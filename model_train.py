import os
import copy
import wandb
import cv2
import PIL.Image as Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from Loss.loss import Model_loss
from Utils.utils import AverageMeter, calculate_psnr
from Model.model import ESPCN
from Dataloader.dataloader import get_data_loader
from Model_config.model_config import Model_Config


class Model_Train:
    def __init__(self, config):
        self.config = config

        if self.config.use_gpu:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            cudnn.benchmark = True
            print(f"Device: {self.device}")
            # Set seed for reproducability
            torch.manual_seed(config.seed)
        else:
            self.device = torch.device("cpu")
            print(f"Device: {self.device}")

        # wandb.init(project="ESPCN")
        # wandb.config.update(self.config)

        self.train_loader, self.val_loader = get_data_loader(dirpath_train=config.dirpath_train,
                                                             dirpath_val=config.dirpath_val,
                                                             scaling_factor=config.scaling_factor,
                                                             patch_size=config.patch_size,
                                                             stride=config.stride)

        for idx, (lr_image, hr_image) in enumerate(self.train_loader):
            print(f"Training - lr_image: {lr_image.shape}, hr_image: {hr_image.shape}")
            print(f"Training - lr_image: {type(lr_image[0])}, hr_image: {type(hr_image[0])}")
            print(f"Training - lr_image: {lr_image[0]}, hr_image: {hr_image[0]}\n")
            break

        for idx, (lr_image, hr_image) in enumerate(self.val_loader):
            print(f"Validation - lr_image: {lr_image.shape}, hr_image: {hr_image.shape}")
            print(f"Validation - lr_image: {type(lr_image[0])}, hr_image: {type(hr_image[0])}")
            print(f"Validation - lr_image: {lr_image[0]}, hr_image: {hr_image[0]}\n")
            break

        self.model = ESPCN(num_channels=1, scaling_factor=config.scaling_factor)
        self.model.to(self.device)

        # wandb.watch(self.model)

        self.criterion = Model_loss()
        self.optimizer = torch.optim.Adam([{'params': self.model.parameters()},
                                           # As per paper, Sec 3.2, The final layer learns 10 times slower
                                           # {
                                           #     'params': self.model.pixelShuffle_last_layer.pixelShuffle_layer.parameters(),
                                           #     'lr': config.learning_rate * 0.1}
                                           ],
                                          lr=config.learning_rate)

    def train(self):
        self.model.train()
        running_loss = AverageMeter()
        for data in self.train_loader:
            inputs, labels = data
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            # print(inputs.shape , labels.shape)
            prediction = self.model(inputs)
            loss = self.criterion(prediction, labels)
            running_loss.update(loss.item(), len(inputs))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return running_loss

    def evaluate(self):
        # Evaluate the Model
        self.model.eval()
        running_psnr = AverageMeter()
        running_loss = AverageMeter()
        for data in self.val_loader:
            inputs, labels = data
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            with torch.no_grad():
                preds = self.model(inputs).clamp(0.0, 1.0)
                loss = self.criterion(preds, labels)
                running_loss.update(loss.item(), len(inputs))
            running_psnr.update(calculate_psnr(preds, labels), len(inputs))
        print('eval psnr: {:.2f}'.format(running_psnr.avg))
        return preds, running_psnr, running_loss

    def model_train(self):
        best_weights = copy.deepcopy(self.model.state_dict())
        best_epoch = 0
        best_psnr = 0.0

        print("Start Model Training ..... ")
        for epoch in tqdm(range(self.config.epochs)):
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.config.learning_rate * (0.1 ** (epoch // int(self.config.epochs * 0.8)))

            training_loss = self.train()
            if not os.path.exists(self.config.dirpath_out):
                os.makedirs(self.config.dirpath_out)
                print("Created")
            torch.save(self.model.state_dict(), os.path.join(self.config.dirpath_out, 'epoch_{}.pth'.format(epoch)))
            preds, running_psnr, validation_loss = self.evaluate()
            if running_psnr.avg > best_psnr:
                best_epoch = epoch
                best_psnr = running_psnr.avg
                best_weights = copy.deepcopy(self.model.state_dict())

            print(
                f"\nEpoch: {epoch}, Training Loss: {training_loss.avg}, PSNR: {running_psnr.avg}, Validation Loss: {validation_loss.avg}\n")

            # wandb.log({"Epoch": epoch, "base_lr": self.config.learning_rate, "sub_pixel_layer_lr": param_group['lr'],
            #            "Training Loss": training_loss.avg, "PSNR": running_psnr.avg,
            #            "Validation Loss": validation_loss.avg})

            if epoch % self.config.log_interval == 0:
                img = preds[0].mul(255.0).cpu().squeeze().numpy()
                # wandb.log({"Upscaled Images": [wandb.Image(img, caption=f"PSNR: {running_psnr.avg}")]})

        print('Best Epoch: {}, PSNR: {:.2f}'.format(best_epoch, best_psnr))
        torch.save(best_weights, os.path.join(self.config.dirpath_out, 'best.pth'))
        return self.model.load_state_dict(best_weights)

    def inferance(self):
        self.model.load_state_dict(torch.load(self.config.fpath_weights))
        self.model.to(self.device)
        self.model.eval()
        hr_image = Image.open(self.config.fpath_image).convert('RGB')
        lr_y, hr_y, ycbcr, bicubic_image = self.prepare_image(hr_image, self.device)
        with torch.no_grad():
            preds = self.model(lr_y)

        psnr_hr_sr = calculate_psnr(hr_y, preds)
        print('PSNR (HR/SR): {:.2f}'.format(psnr_hr_sr))
        #print(preds.mul(255.0).cpu().numpy().shape)

        preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)
        print(preds.shape, ycbcr[..., 1].shape, ycbcr[..., 2].shape)

        output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
        output = np.clip(cv2.cvtColor(output, cv2.COLOR_YCrCb2RGB), 0.0, 255.0).astype(np.uint8)
        output = Image.fromarray(output)

        output.save(os.path.join(self.config.dirpath_out,os.path.basename(self.config.fpath_image).replace(".png", f"_espcn_x{self.config.scaling_factor}.png")))

        # Plot Image Comparison
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(100, 100))
        ax1.imshow(np.array(hr_image))
        ax1.set_title("HR Image")
        ax2.imshow(bicubic_image)
        ax2.set_title("Bicubic Image x3")
        ax3.imshow(np.array(output))
        ax3.set_title("SR Image x3 (PSNR: {:.2f} dB)".format(psnr_hr_sr))
        fig.suptitle('ESPCN Single Image Super Resolution')
        plt.show()
        fig.set_size_inches(50, 50, forward=True)
        if not os.path.exists(self.config.results_output):
            os.makedirs(self.config.results_output)
            print("load data")
        fig.savefig(os.path.join(self.config.results_output, "result.png"), dpi=100)

    def prepare_image(self, hr_image, device):
        # Load HR image: rH x rW x C, r: scaling factor
        hr_width = (hr_image.width // self.config.scaling_factor) * self.config.scaling_factor
        hr_height = (hr_image.height // self.config.scaling_factor) * self.config.scaling_factor
        hr_image = hr_image.resize((hr_width, hr_height), resample=Image.BICUBIC)

        # LR Image: H x W x C
        # As in paper, Sec. 3.2: sub-sample images by up-scaling factor
        lr_image = hr_image.resize((hr_image.width // self.config.scaling_factor, hr_image.height // self.config.scaling_factor),resample=Image.BICUBIC)

        # Generate Bicubic image for performance comparison
        bicubic_image = lr_image.resize((lr_image.width * self.config.scaling_factor, lr_image.height * self.config.scaling_factor),resample=Image.BICUBIC)
        bicubic_image.save(os.path.join(self.config.dirpath_out,os.path.basename(self.config.fpath_image).replace(".png", f"_bicubic_x{self.config.scaling_factor}.png")))

        # Convert PIL image to numpy array
        hr_image = np.array(hr_image).astype(np.float32)
        lr_image = np.array(lr_image).astype(np.float32)
        bicubic_image = np.array(bicubic_image).astype(np.float32)

        # Convert RGB to YCbCr
        hr_image = cv2.cvtColor(hr_image, cv2.COLOR_RGB2YCrCb)
        lr_image = cv2.cvtColor(lr_image, cv2.COLOR_RGB2YCrCb)
        bicubic_image_ycrcb = cv2.cvtColor(bicubic_image, cv2.COLOR_RGB2YCrCb)

        # As per paper, using only the luminescence channel gave the best outcome
        hr_y = hr_image[:, :, 0]
        lr_y = lr_image[:, :, 0]

        # Normalize images
        hr_y /= 255.
        lr_y /= 255.
        bicubic_image /= 255.
        # Convert Numpy to Torch Tensor and send to device
        hr_y = torch.from_numpy(hr_y).to(device)
        hr_y = hr_y.unsqueeze(0).unsqueeze(0)
        lr_y = torch.from_numpy(lr_y).to(device)
        lr_y = lr_y.unsqueeze(0).unsqueeze(0)
        return lr_y, hr_y, bicubic_image_ycrcb, bicubic_image



