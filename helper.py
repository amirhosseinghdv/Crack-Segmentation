# Dataset class

# Imports
import os
import shutil
import cv2
import numpy as np
from PIL import Image
import gdown
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow

import torch
import albumentations as albu
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset

os.makedirs('/content/outputs', exist_ok=True)



model = torch.load('/content/model.pth')

class Dataset(BaseDataset):

    CLASSES = ['asphalt', 'cracks']

    def __init__(
            self,
            images_dir,
            # masks_dir,
            classes=None,
            augmentation=None,
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        # self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]

        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # mask = cv2.imread(self.masks_fps[i], 0)

        # extract certain classes from mask (e.g. cracks)
        # masks = [(mask == v) for v in self.class_values]
        # mask = np.stack(masks, axis=-1).astype('float')

        # apply augmentations
        if self.augmentation:
            # sample = self.augmentation(image=image, mask=mask)
            # image, mask = sample['image'], sample['mask']
            sample = self.augmentation(image=image)
            image = sample['image']


        # apply preprocessing
        if self.preprocessing:
            # sample = self.preprocessing(image=image, mask=mask)
            # image, mask = sample['image'], sample['mask']
            sample = self.preprocessing(image=image)
            image = sample['image']



        # return image, mask
        return image


    def __len__(self):
        return len(self.ids)








def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        # albu.PadIfNeeded(384, 480)
        albu.PadIfNeeded(288, 512)
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):

    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)








# Selecting model and encoder
ENCODER = 'mit_b5'
ENCODER_WEIGHTS = 'imagenet'



aaa=100
try:
    preprocessing_fn
except NameError:
    aaa=1
if aaa==100:
    del preprocessing_fn


preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)





aaa=100
try:
    test_dataset
except NameError:
    aaa=1
if aaa==100:
    del test_dataset


# create test dataset
test_dataset = Dataset(
    '/content/Crack-Segmentation/samples',
    augmentation=get_validation_augmentation(),
    preprocessing=get_preprocessing(preprocessing_fn),
    classes=['cracks']
)



aaa=100
try:
    test_dataloader
except NameError:
    aaa=1
if aaa==100:
    del test_dataloader

test_dataloader = DataLoader(test_dataset)




aaa=100
try:
    test_dataset_vis
except NameError:
    aaa=1
if aaa==100:
    del test_dataset_vis


# test dataset without transformations for image visualization
test_dataset_vis = Dataset(
    '/content/Crack-Segmentation/samples',
    classes=['cracks']
)




# helper function for data visualization
def visualize(indexx, **images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.savefig('/content/outputs/' + str(indexx))



def inference():

    for i in range(len(test_dataset)):

        image_vis = test_dataset_vis[i].astype('uint8')
        image = test_dataset[i]

        x_tensor = torch.from_numpy(image).to('cuda').unsqueeze(0)


        pr_mask = model.predict(x_tensor)
        pr_mask = (pr_mask.squeeze().cpu().numpy().round())

        visualize(
            indexx = os.listdir('/content/Crack-Segmentation/samples')[i],
            image=image_vis,
            predicted_mask=pr_mask            
        )





if __name__ == "__main__":
    inference()












