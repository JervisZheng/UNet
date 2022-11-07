import cv2
import numpy as np


def image_fill(image, image_outsize):
    h, w, _ = image.shape
    new_images = np.zeros((image_outsize, image_outsize, 3), dtype=np.uint8)

    ratio = image_outsize / max(h, w)  # 等比缩放比例

    image_ = cv2.resize(image, (int(w * ratio), int(h * ratio)))

    new_h, new_w, _ = image_.shape
    (y, x) = ((image_outsize - new_h) // 2, (image_outsize - new_w) // 2)

    new_images[y:y+new_h, x:x+new_w, :] = image_

    return new_images


if __name__ == '__main__':
    image = cv2.imread(r'../1.png')
    background = image_fill(image, 640)
    cv2.imshow('1', background)
    cv2.waitKey()
