from models.morn import MORN
import torch
from torchsummary import summary
from models.asrn_res import ResNet
from models.moran import MORAN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # PyTorch v0.4.0

if __name__ == '__main__':
    # resnet = ResNet(1)
    # morn = MORN(1, 32, 100)
    # resnet.to(device)
    # morn.to(device)
    # summary(resnet, (1, 32, 100))
    # summary(morn, (1, 32, 100))
    # 1, 16, 256, 32, 100,
    # text = torch.LongTensor(opt.batchSize * 5)
    # text_rev = torch.LongTensor(opt.batchSize * 5)
    # length = torch.IntTensor(opt.batchSize)
    moran = MORAN(1, 16, 256, 32, 100, BidirDecoder=True, CUDA=True)
    moran.to(device)
    summary(moran, (1, 32, 100))
    pass
