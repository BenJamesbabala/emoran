import torch.nn as nn
from models.morn import MORN
from models.asrn_res import ASRN
from stn.mnist_model_summuy import stn
import torch


class MORAN(nn.Module):

    def __init__(self, nc, nclass, nh, targetH, targetW, BidirDecoder=False,
                 inputDataType='torch.cuda.FloatTensor', maxBatch=256, CUDA=True):
        super(MORAN, self).__init__()
        self.MORN = MORN(nc, targetH, targetW, inputDataType, maxBatch, CUDA)
        self.ASRN = ASRN(targetH, nc, nclass, nh, BidirDecoder, CUDA)
        self.STN = stn()

    def forward(self, x, length=torch.IntTensor(16), text=torch.LongTensor(16 * 5),
                text_rev=torch.LongTensor(16 * 5), test=False,
                debug=False):
        if debug:
            x = self.STN(x)
            x_rectified, demo = self.MORN(x, test, debug=debug)
            preds = self.ASRN(x_rectified, length, text, text_rev, test)
            return preds, demo
        else:
            x = self.STN(x)
            x_rectified = self.MORN(x, test, debug=debug)
            preds = self.ASRN(x_rectified, length, text, text_rev, test)
            return preds
