import torch
import argparse
from models.UNet import UNet
from torch.utils.data import DataLoader
from data.dataset import HumanSegDataset
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


class Train:
    def __init__(self, opt):
        self.device = opt.device
        self.writer = SummaryWriter('logs')
        self.weights = opt.weights
        self.data = opt.data
        self.batch_size = opt.batch_size
        self.epochs = opt.epochs
        self.image_size = opt.image_size
        self.save_path = opt.save
        if self.weights:
            self.model = UNet().load_state_dict(torch.load(self.weights)).to(self.device)
        else:
            self.model = UNet().to(self.device)
        self.train_loader = DataLoader(HumanSegDataset(self.data, self.image_size),
                                       batch_size=self.batch_size, shuffle=True)
        # 定义损失函数
        if opt.loss_func == 'BCE':
            self.loss_func = torch.nn.BCELoss()
        elif opt.loss_func:
            self.loss_func = torch.nn.MSELoss()

        # 定义优化器
        if opt.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters())
        elif opt.optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(self.model.parameters())

    def __call__(self, *args, **kwargs):
        for epoch in range(self.epochs):
            train_sum_loss = 0.
            for image_, target_ in tqdm(self.train_loader):
                image, target = image_.to(self.device), target_.to(self.device)

                pre = self.model(image)

                loss = self.loss_func(pre, target)
                train_sum_loss += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            train_loss = train_sum_loss / len(self.train_loader)
            print(f'epoch-{epoch}--train_loss--->{train_loss}')
            self.writer.add_scalar('train_loss', train_loss, epoch)
            torch.save(self.model.state_dict(), self.save_path)


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='', help='initial weights path')
    parser.add_argument('--data', type=str, default=r'E:\Edgedownloads\humanseg', help='data path')
    # parser.add_argument('--model', type=str, default='', help='model path')
    parser.add_argument('--epochs', type=int, default=300, help='total training epochs')
    parser.add_argument('--batch_size', type=int, default=6, help='total batch size')
    parser.add_argument('--image-size', type=int, default=256, help='train image size')
    parser.add_argument('--device', type=str, default='cuda', help='device: cuda or cpu')
    parser.add_argument('--optimizer', type=str, choices=['Adam', 'SGD'], default='Adam', help='optimizer')
    parser.add_argument('--loss_func', type=str, choices=['MSE', 'BCE'], default='MSE', help='loss function')
    parser.add_argument('--save', type=str, default='./weights/params.pt', help='save path')
    return parser.parse_known_args()[0] if known else parser.parse_args()


if __name__ == '__main__':
    opt = parse_opt()
    train = Train(opt)
    train()

