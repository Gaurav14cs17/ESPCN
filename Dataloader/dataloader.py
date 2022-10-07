import os
import random
from glob import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torch.utils.data import IterableDataset, DataLoader


# Ref.: https://github.com/yjn870/ESPCN-pytorch/blob/ab84ee1bccb2978f2f9b88f9e0315d9be12c099e/prepare.py#L16


class SRTrainDataset(IterableDataset):
    def __init__(self, dirpath_images, scaling_factor, patch_size, stride):
        """ Training Dataset
        :param dirpath_images: path to training images directory
        :param scaling_factor: up-scaling factor to use, default = 3
        :param patch_size: size of sub-images to be extracted
        :param stride: stride used for extracting sub-images
        """

        self.dirpath_images = dirpath_images
        self.scaling_factor = scaling_factor
        self.patch_size = patch_size
        self.stride = stride

    def preprocess(self, image):
        image = np.array(image).astype(np.float32)
        # Convert BGR to YCbCr
        image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        # As per paper, using only the luminescence channel gave the best outcome
        image = image[:, :, 0]

        return image

    def __iter__(self):
        for fpath_image in glob(os.path.join(self.dirpath_images, "*.jpg")):
            hr_image = Image.open(fpath_image)
            hr_image = hr_image.convert('RGB')
            hr_image_width, hr_image_height = hr_image.width, hr_image.height
            hr_width = (hr_image_width // self.scaling_factor) * self.scaling_factor
            hr_height = (hr_image_height // self.scaling_factor) * self.scaling_factor
            ##########################################################################

            # Resize Image
            hr_image = hr_image.resize((hr_width, hr_height), resample=Image.BICUBIC)
            # LR Image: H x W x C
            # As in paper, Sec. 3.2: sub-sample images by up-scaling factor
            lr_image = hr_image.resize((hr_width // self.scaling_factor, hr_height // self.scaling_factor),
                                       resample=Image.BICUBIC)

            hr_image = self.preprocess(hr_image)
            lr_image = self.preprocess(lr_image)
            rows, cols = lr_image.shape[0:2]

            for i in range(0, rows - self.patch_size + 1, self.stride):
                for j in range(0, cols - self.patch_size + 1, self.stride):
                    # lr_crop: w = 17, h = 17
                    lr_crop = lr_image[i:i + self.patch_size, j:j + self.patch_size]
                    # hr_crop: w = 17 * r, h = 17 * r
                    hr_crop = hr_image[
                              i * self.scaling_factor:i * self.scaling_factor + self.patch_size * self.scaling_factor,
                              j * self.scaling_factor:j * self.scaling_factor + self.patch_size * self.scaling_factor
                              ]

                    lr_crop = np.expand_dims(lr_crop / 255.0, axis=0)
                    hr_crop = np.expand_dims(hr_crop / 255.0, axis=0)
                    yield lr_crop, hr_crop

    def __len__(self):
        return len(self.all_images)


# Valid Dataset Class
class SRValidDataset(IterableDataset):
    def __init__(self, dirpath_images, scaling_factor):
        """ Validation Dataset
        :param dirpath_images: path to validation images directory
        :param scaling_factor: up-scaling factor to use, default = 3
        """
        self.dirpath_images = dirpath_images
        self.scaling_factor = scaling_factor

    def preprocess(self, image):
        image = np.array(image).astype(np.float32)
        # Convert BGR to YCbCr
        image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        # As per paper, using only the luminescence channel gave the best outcome
        image = image[:, :, 0]
        image = np.expand_dims(image / 255.0, axis=0)
        return image

    def __iter__(self):
        for fpath_image in glob(os.path.join(self.dirpath_images, "*.jpg")):
            # Load HR image: rH x rW x C, r: scaling factor
            hr_image = Image.open(fpath_image).convert('RGB')
            hr_width = (hr_image.width // self.scaling_factor) * self.scaling_factor
            hr_height = (hr_image.height // self.scaling_factor) * self.scaling_factor
            ###############################################################################
            hr_image = hr_image.resize((hr_width, hr_height), resample=Image.BICUBIC)
            # LR Image: H x W x C
            # As in paper, Sec. 3.2: sub-sample images by up-scaling factor
            lr_image = hr_image.resize((hr_width // self.scaling_factor, hr_height // self.scaling_factor),
                                       resample=Image.BICUBIC)
            lr_y = self.preprocess(lr_image)
            hr_y = self.preprocess(hr_image)
            yield lr_y, hr_y

    def __len__(self):
        return len(self.all_images)


# Ref.: https://discuss.pytorch.org/t/how-to-shuffle-an-iterable-dataset/64130/6
class ShuffleDataset(IterableDataset):
    def __init__(self, dataset, buffer_size):
        super().__init__()
        self.dataset = dataset
        self.buffer_size = buffer_size

    def __iter__(self):
        shufbuf = []
        try:
            dataset_iter = iter(self.dataset)
            for i in range(self.buffer_size):
                shufbuf.append(next(dataset_iter))
        except:
            self.buffer_size = len(shufbuf)

        try:
            while True:
                try:
                    item = next(dataset_iter)
                    evict_idx = random.randint(0, self.buffer_size - 1)
                    yield shufbuf[evict_idx]
                    shufbuf[evict_idx] = item
                except StopIteration:
                    break
            while len(shufbuf) > 0:
                yield shufbuf.pop()
        except GeneratorExit:
            pass


def get_data_loader(dirpath_train, dirpath_val, scaling_factor, patch_size, stride):
    """ Function to return train/val data loader
    :param dirpath_train (str): path to directory containing high resolution training images
    :param dirpath_val (str): path to directory containing high resolution validation images
    :param scaling_factor (int): Number by which to scale the lr image to hr image
    :param patch_size (int): size of sub-images extracted from original images
    :param stride (int): sub-images extraction stride
    :return (torch.utils.data.DataLoader): training and validation data loader
    """
    # As per paper, Sec. 3.2, sub-images are extracted only during the training phase

    dataset = SRTrainDataset(dirpath_images=dirpath_train, scaling_factor=scaling_factor, patch_size=patch_size,stride=stride)
    train_dataset = ShuffleDataset(dataset, 1024)

    train_loader = DataLoader(train_dataset, batch_size=16, num_workers=4, pin_memory=False)
    valid_dataset = SRValidDataset(dirpath_images=dirpath_val, scaling_factor=scaling_factor)
    val_loader = DataLoader(valid_dataset, batch_size=1, pin_memory=False)
    return train_loader, val_loader


if __name__ == '__main__':
    # Test DataLoaders
    train_loader, val_loader = get_data_loader(dirpath_train="/media/gaurav/DATA/labs/ESPCN/dataset/train",
                                               dirpath_val="/media/gaurav/DATA/labs/ESPCN/dataset/val",
                                               scaling_factor=3,
                                               patch_size=17,
                                               stride=13)

    for idx, (lr_image, hr_image) in enumerate(train_loader):
        print(f"Training - lr_image: {lr_image.shape}, hr_image: {hr_image.shape}")
        break

    for idx, (lr_image, hr_image) in enumerate(val_loader):
        print(f"Validation - lr_image: {lr_image.shape}, hr_image: {hr_image.shape}")
        break

    for idx, (lr_image, hr_image) in enumerate(train_loader):
        print(f"lr_image: {lr_image.shape}, hr_image: {hr_image.shape}")
        lr = lr_image[0].numpy().transpose(1, 2, 0)
        hr = hr_image[0].numpy().transpose(1, 2, 0)
        print(f"{lr.shape}, {hr.shape}")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        ax1.imshow(lr)
        ax1.set_title("Low Res")
        ax2.imshow(hr)
        ax2.set_title("High Res")
        plt.show()
        break

    for idx, (lr_image, hr_image) in enumerate(val_loader):
        print(f"lr_image: {lr_image.shape}, hr_image: {hr_image.shape}")
        lr = lr_image[0].numpy().transpose(1, 2, 0)
        hr = hr_image[0].numpy().transpose(1, 2, 0)
        print(f"{lr.shape}, {hr.shape}")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        ax1.imshow(lr)
        ax1.set_title("Low Res")
        ax2.imshow(hr)
        ax2.set_title("High Res")
        plt.show()
        break
