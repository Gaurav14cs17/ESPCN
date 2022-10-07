import torch.nn as nn


class Model_loss(nn.Module):
    def __init__(self):
        super(Model_loss, self).__init__()
        self.model_loss = nn.MSELoss()

    def forward(self, pred, label):
        loss = self.model_loss(pred, label)
        return loss
