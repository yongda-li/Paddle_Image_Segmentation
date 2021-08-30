import cv2
import numpy as np
import matplotlib.pyplot as plt

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, image, label=None):
        for t in self.transforms:
            image, label = t(image, label)
        return image, label


class Normalize(object):
    def __init__(self, mean_val, std_val, val_scale=1):
        # set val_scale = 1 if mean and std are in range (0,1)
        # set val_scale to other value, if mean and std are in range (0,255)
        self.mean = np.array(mean_val, dtype=np.float32)
        self.std = np.array(std_val, dtype=np.float32)
        self.val_scale = 1/255.0 if val_scale==1 else 1
    def __call__(self, image, label=None):
        image = image.astype(np.float32)
        image = image * self.val_scale
        image = image - self.mean
        image = image * (1 / self.std)
        return image, label


class ConvertDataType(object):
    def __call__(self, image, label=None):
        if label is not None:
            label = label.astype(np.int64)
        return image.astype(np.float32), label


class Pad(object):
    def __init__(self, size, ignore_label=255, mean_val=0, val_scale=1):
        # set val_scale to 1 if mean_val is in range (0, 1)
        # set val_scale to 255 if mean_val is in range (0, 255) 
        factor = 255 if val_scale == 1 else 1

        self.size = size
        self.ignore_label = ignore_label
        self.mean_val=mean_val
        # from 0-1 to 0-255
        if isinstance(self.mean_val, (tuple,list)):
            self.mean_val = [int(x* factor) for x in self.mean_val]
        else:
            self.mean_val = int(self.mean_val * factor)


    def __call__(self, image, label=None):
        h, w, c = image.shape
        pad_h = max(self.size - h, 0)
        pad_w = max(self.size - w, 0)

        pad_h_half = int(pad_h / 2)
        pad_w_half = int(pad_w / 2)

        if pad_h > 0 or pad_w > 0:

            image = cv2.copyMakeBorder(image,
                                       top=pad_h_half,
                                       left=pad_w_half,
                                       bottom=pad_h - pad_h_half,
                                       right=pad_w - pad_w_half,
                                       borderType=cv2.BORDER_CONSTANT,
                                       value=self.mean_val)
            if label is not None:
                label = cv2.copyMakeBorder(label,
                                           top=pad_h_half,
                                           left=pad_w_half,
                                           bottom=pad_h - pad_h_half,
                                           right=pad_w - pad_w_half,
                                           borderType=cv2.BORDER_CONSTANT,
                                           value=self.ignore_label)
        return image, label


# TODO
class CenterCrop(object):
    def __init__(self, crop_size):
        self.crop_h = crop_size
        self.crop_w = crop_size
        
    def __call__(self, image, label=None):
        h, w, c = image.shape
        top = (h - self.crop_h) // 2
        left = (w - self.crop_w) // 2
        image = image[top:top+self.crop_h, left:left+self.crop_w, :]
        if label is not None:
            label = label[top:top+self.crop_h, left:left+self.crop_w, :]
        return image, label


# TODO
class Resize(object):
    def __init__(self, size=256):
        self.size = size
    
    def __call__(self, image, label=None):
        image = cv2.resize(image, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        if label is not None:
            label = cv2.resize(label, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        return image, label

# TODO
class RandomFlip(object):
    def __call__(self, image, label=None):
        prob_flip = np.random.rand()
        if prob_flip > 0.5:
            image = cv2.flip(image, 0)
            if label is not None:
                label = cv2.flip(label, 0)
        return image, label


# # TODO
class RandomCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, image, label=None):
        h,w,c = image.shape
        top = np.random.uniform(h - self.crop_size)
        left = np.random.uniform(w - self.crop_size)

        assert top >= 0, "Error: crop_size > image height"
        assert left >= 0, "Error: crop_size > image width"

        rect = np.array([int(top),
                        int(left),
                        int(top+self.crop_size),
                        int(left+self.crop_size)
        ])

        image = image[rect[0]:rect[2], rect[1]:rect[3],:]
        if label is not None:
            label = label[rect[0]:rect[2], rect[1]:rect[3],:]
        return image, label




# # TODO
class Scale(object):
    def __call__(self, image, label=None, scale=1.0):
        if not isinstance(scale, (list, tuple)):
            scale = [scale, scale]
        h, w, c = image.shape
        image = cv2.resize(image,
                            (int(h*scale[0]), int(w*scale[1])),
                            interpolation=cv2.INTER_LINEAR
                            )
        if label is not None:
            label = cv2.resize(label,
                                (int(h*scale[0]),int(w*scale[1])),
                                interpolation=cv2.INTER_LINEAR
                            )
        return image, label


# # TODO
class RandomScale(object):
    def __init__(self, min_scale=0.5, max_scale=2.0, step=0.25):
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.step = step
        self.scale = Scale()

    def __call__(self, image, label=None):
        if self.step == 0:
            self.random_scale = np.random.uniform(self.min_scale, self.max_scale, 1)[0]
        else:
            num_steps = int((self.max_scale-self.min_scale)/self.step + 1)
            scale_factors = np.linspace(self.min_scale, self.max_scale, num_steps)
            np.random.shuffle(scale_factors)
            self.random_scale = scale_factors[0]
        image, label = self.scale(image, label, scale=self.random_scale)
        return image, label



def main():
    image = cv2.imread('./work/dummy_data/JPEGImages/2008_000064.jpg')
    label = cv2.imread('./work/dummy_data/GroundTruth_trainval_png/2008_000064.png')

    # TODO: crop_size
    crop_size = 256
    
    # TODO: Transform: RandomSacle, RandomFlip, Pad, RandomCrop
    # Pad
    pad = Pad(crop_size, mean_val=[0.485,0.412,0.788]) 
    pad_img, pad_label = pad(image, label=label)
    cv2.imwrite('pad.png', pad_img)

    # CenterCrop
    center_crop = CenterCrop(crop_size)
    cnt_img, cnt_label = center_crop(image, label=label)
    cv2.imwrite('cnt_crop.png', cnt_img)

    # Resize
    res = Resize(crop_size)
    res_img, res_label = res(image, label)
    cv2.imwrite('resize.png', res_img)

    # RandomClip
    rand_flip = RandomFlip()
    flip_img, flip_label = rand_flip(image, label)
    cv2.imwrite('random_flip.png', flip_img)

    # RandomCrop
    rand_crop = RandomCrop(crop_size)
    rnd_crop_img, rnd_crop_label = rand_crop(image, label)
    cv2.imwrite('random_crop.png', rnd_crop_img)

    # Scale
    scale = Scale()
    scale_img, scale_label = scale(image, label, 1.5)
    cv2.imwrite('scale.png', scale_img)

    # RandomScale
    rand_scale = RandomScale(min_scale=1.5, max_scale=3.0, step=0.5)
    rand_scale_image, rand_scale_label = rand_scale(image, label)
    cv2.imwrite('rand_scale.png', rand_scale_image)
    # for i in range(10):
        # TODO: call transform
        # TODO: save image

if __name__ == "__main__":
    main()
