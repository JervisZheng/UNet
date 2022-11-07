import torch
from torch import nn
from models.UNet import UNet
from utils.imagefill import image_fill
import argparse
from torchvision import transforms
from torchvision.utils import save_image
from pathlib import Path
import cv2


class Detect(nn.Module):
    def __init__(self, opt):
        super(Detect, self).__init__()
        self.images_size = opt.image_size
        self.weight = opt.weights
        self.device = opt.device
        self.model = UNet().to(self.device)
        self.data = opt.data
        self.result = opt.result
        self.model.load_state_dict(torch.load(self.weight, map_location=self.device))

    def forward(self, x):
        self.detect(x)

    def detect(self, data_path):
        path = Path(data_path)
        for image in path.iterdir():
            image_name = image.name
            image = cv2.imread(str(image))
            image = image_fill(image, self.images_size)
            image = transforms.ToTensor()(image)
            image = image.unsqueeze(dim=0).to(self.device)
            out = self.model(image)
            save_image(out, self.result + '/' + image_name)


def run():
    opt = opt_parse()
    detect = Detect(opt)
    detect(opt.data)


def opt_parse(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=r'./weights/params.pt', help='initial weights path')
    parser.add_argument('--data', type=str, default=r'./testdata', help='data path')
    parser.add_argument('--image_size', type=int, default=256, help='train image size')
    parser.add_argument('--device', type=str, default='cpu', help='device: cuda or cpu')
    parser.add_argument('--result', type=str, default='./result', help='result')
    return parser.parse_known_args()[0] if known else parser.parse_args()


if __name__ == '__main__':
    run()
