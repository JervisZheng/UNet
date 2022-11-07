from torch.utils.data import Dataset
from pathlib import Path
import cv2
from torchvision import transforms
from utils.imagefill import image_fill

my_transforms = transforms.Compose([
    transforms.ToTensor()
])


class HumanSegDataset(Dataset):
    def __init__(self, rootpath, image_size):
        self.rootpath = Path(rootpath)
        self.datas = []
        self.image_size = image_size
        images_path = self.rootpath / 'images'
        labels_path = self.rootpath / 'labels'
        for image in images_path.iterdir():
            label_name = image.name[:str(image.name).rfind('.')] + '.png'
            label = labels_path / label_name
            self.datas.append((image, label))

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, item):
        data = self.datas[item]

        image = cv2.imread(str(data[0]))
        image = image_fill(image, self.image_size)
        label = cv2.imread(str(data[1]))
        label = image_fill(label, self.image_size)

        return my_transforms(image), my_transforms(label)


if __name__ == '__main__':
    data = HumanSegDataset(r'E:\Edgedownloads\humanseg', 640)
