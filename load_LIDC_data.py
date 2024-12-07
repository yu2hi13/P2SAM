import os
import random
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy import ndimage
from scipy.ndimage import zoom
from einops import repeat

def random_rot_flip(image, label):
    # Randomly flip the image and label horizontally
    if np.random.rand() < 0.5:
        image = np.flip(image, axis=(1, 2))
        label = np.flip(label, axis=1)

    # Randomly flip the image and label vertically
    if np.random.rand() < 0.5:
        image = np.flip(image, axis=(0, 2))
        label = np.flip(label, axis=0)

    # Randomly rotate the image and label by 0, 90, 180, or 270 degrees
    rots = np.random.randint(0, 4)
    image = np.rot90(image, rots, axes=(1, 2))
    label = np.rot90(label, rots, axes=(0, 1))

    return image, label

def random_rotate(image, label):
    # Randomly rotate the image and label by an angle between -20 and 20 degrees
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label

class RandomGenerator(object):
    def __init__(self, output_size, low_res, test=False):
        self.output_size = output_size
        self.low_res = low_res
        self.test = test

    def __call__(self, sample):
        image, label, label_four = sample['image'], sample['label'], sample['label_four']
        label_four = np.stack(label_four, axis=0).astype(np.int64)
        label = label.squeeze()

        if not self.test:
            image, label = random_rot_flip(image, label)

        image_oc = image.copy()
        x, y = image.shape[-2:]

        # Resize image and label to the specified output size
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)

        label_h, label_w = label.shape
        low_res_label = zoom(label, (self.low_res[0] / label_h, self.low_res[1] / label_w), order=0)

        # Convert numpy arrays to torch tensors
        image = torch.from_numpy(image.astype(np.float32))
        image = repeat(image, 'c h w -> (repeat c) h w', repeat=3)
        label = torch.from_numpy(label.astype(np.float32))
        low_res_label = torch.from_numpy(low_res_label.astype(np.float32))

        sample = {
            'image': image,
            'label': label.long(),
            'low_res_label': low_res_label.long(),
            'image_oc': image_oc,
            'label_four': label_four
        }
        return sample

class LIDC_IDRI(Dataset):
    def __init__(self, dataset_location, transform=None):
        self.transform = transform
        self.images = []
        self.labels = []
        self.series_uid = []

        max_bytes = 2**31 - 1
        data = {}

        # Load data from pickle files
        for file in os.listdir(dataset_location):
            filename = os.fsdecode(file)
            if '.pickle' in filename:
                print("Loading file", filename)
                file_path = os.path.join(dataset_location, filename)
                bytes_in = bytearray(0)
                input_size = os.path.getsize(file_path)
                with open(file_path, 'rb') as f_in:
                    for _ in range(0, input_size, max_bytes):
                        bytes_in += f_in.read(max_bytes)
                new_data = pickle.loads(bytes_in)
                data.update(new_data)

        for key, value in data.items():
            self.images.append(value['image'].astype(float))
            self.labels.append(value['masks'])
            self.series_uid.append(value['series_uid'])

        assert len(self.images) == len(self.labels) == len(self.series_uid)

        for img in self.images:
            assert np.max(img) <= 1 and np.min(img) >= 0
        for label in self.labels:
            assert np.max(label) <= 1 and np.min(label) >= 0

        del new_data
        del data

    def __getitem__(self, index):
        image = np.expand_dims(self.images[index], axis=0)
        # Randomly select one of the four labels for this image
        label = self.labels[index][random.randint(0, 3)].astype(float)
        label_four = self.labels[index]
        while label.sum() == 0:
            label = self.labels[index][random.randint(0, 3)].astype(float)

        # Convert image and label to torch tensors
        image = torch.from_numpy(image).type(torch.FloatTensor)
        label = torch.from_numpy(label).type(torch.LongTensor).unsqueeze(0)

        image = np.array(image)
        label = np.array(label)

        sample = {'image': image, 'label': label, 'label_four': label_four}
        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.images)