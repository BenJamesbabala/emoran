import torch
from torch.autograd import Variable
import tools.utils as utils
import tools.dataset as dataset
from PIL import Image
from collections import OrderedDict
# import cv2
from models.moran import MORAN

model_path = 'demo.pth'
img_path = './demo/1.png'
alphabet = '0:1:2:3:4:5:6:7:8:9:a:b:c:d:e:f:g:h:i:j:k:l:m:n:o:p:q:r:s:t:u:v:w:x:y:z:$'

cuda_flag = False
# print(torch.cuda.is_available())
# if torch.cuda.is_available():
#     cuda_flag = True
#     MORAN = MORAN(1, len(alphabet.split(':')), 256, 32, 100, BidirDecoder=True,
#                   CUDA=cuda_flag)
#     MORAN = MORAN.cuda()
# else:
#     MORAN = MORAN(1, len(alphabet.split(':')), 256, 32, 100, BidirDecoder=True, inputDataType='torch.FloatTensor',
#                   CUDA=cuda_flag)
MORAN = MORAN(1, len(alphabet.split(':')), 256, 32, 100, BidirDecoder=True, inputDataType='torch.FloatTensor',
              CUDA=cuda_flag)

print('loading pretrained model from %s' % model_path)
if cuda_flag:
    state_dict = torch.load(model_path)
else:
    state_dict = torch.load(model_path, map_location='cpu')
MORAN_state_dict_rename = OrderedDict()
for k, v in state_dict.items():
    name = k.replace("module.", "")  # remove `module.`
    MORAN_state_dict_rename[name] = v
MORAN.load_state_dict(MORAN_state_dict_rename)

for p in MORAN.parameters():
    p.requires_grad = False
MORAN.eval()

# load image
converter = utils.strLabelConverterForAttention(alphabet, ':')
transformer = dataset.resizeNormalize((100, 32))


def demo(image_path):
    image = Image.open(image_path).convert('L')
    image = transformer(image)

    if cuda_flag:
        image = image.cuda()
    image = image.view(1, *image.size())
    image = Variable(image)
    text = torch.LongTensor(1 * 5)
    length = torch.IntTensor(1)
    text = Variable(text)
    length = Variable(length)

    max_iter = 20
    t, l = converter.encode('0' * max_iter)
    utils.loadData(text, t)
    utils.loadData(length, l)
    output = MORAN(image, length, text, text, test=True, debug=True)

    preds, preds_reverse = output[0]
    demo = output[1]

    _, preds = preds.max(1)
    _, preds_reverse = preds_reverse.max(1)

    sim_preds = converter.decode(preds.data, length.data)
    sim_preds = sim_preds.strip().split('$')[0]
    sim_preds_reverse = converter.decode(preds_reverse.data, length.data)
    sim_preds_reverse = sim_preds_reverse.strip().split('$')[0]
    print(image_path)
    print('\nResult:\n' + 'Left to Right: ' + sim_preds + '\nRight to Left: ' + sim_preds_reverse)
    return sim_preds


def bijiao(str1="", str2=""):
    str1 = str1.lower()
    str2 = str2.lower()
    for x, y in zip(str1, str2):
        if x != y:
            print(x + "<<<<<error>>>>>>" + y)
            return 0
    return 1


if __name__ == '__main__':
    import os
    from os.path import join

    i = 0
    j = 0

    path = "/home/lz/下载/data_clean/svt-perspective"
    gt = "gt.txt"
    with open(join(path, gt), "r") as f:
        all = f.readlines()
    for name in all:
        name = name.strip("\n").split(' ', 1)
        predict = demo(join(path, name[0]))
        print(name[1])

        j += bijiao(name[1], predict)
        i += 1
        print('正确的数量>>>>', j, "\n\n\n")
    print("acc>>>>", j / i)
    # demo(img_path)
    # cv2.imshow("demo", demo)
    # cv2.waitKey()

    # cute 82.3%
    # ic13 94.04%
    # ic15 all 71.738%
    # ic15 1811 0.778575%
    # svt 86%
    # svt-p 78.6%
