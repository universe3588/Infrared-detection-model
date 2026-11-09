from torch import nn

import os
from loss import SoftIoULoss
from model import *

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


class Net(nn.Module):
    def __init__(self, model_name):
        super(Net, self).__init__()

        self.model_name = model_name
        self.cal_loss = SoftIoULoss()

        if model_name == 'L2SKNet_UNet':
            self.model = L2SKNet_UNet()
        elif model_name == 'L2SKNet_FPN':
            self.model = L2SKNet_FPN()

        elif model_name == 'L2SKNet_1D_UNet':
            self.model = L2SKNet_1D_UNet()
        elif model_name == 'L2SKNet_1D_FPN':
            self.model = L2SKNet_1D_FPN()

    def forward(self, img):
        return self.model(img)

    def loss(self, pred, gt_mask):
        loss = self.cal_loss(pred, gt_mask)
        return loss
