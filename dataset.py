import torch
import numpy as np
from imgaug import augmenters as iaa


class Dataset():
    def __init__(self, samples, training=False, test=False):
        self.samples = samples
        self.training = training
        self.test = test

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        if self.test:
            image, filename = self.samples[item]
        else:
            image, label = self.samples[item]

        if self.training:
            aug = iaa.SomeOf((0, 4),
                               [iaa.Fliplr(p=0.5),
                                iaa.Flipud(p=0.5),
                                iaa.GammaContrast(1.2),
                                iaa.PerspectiveTransform(scale=0.075)])
            image = aug.augment_image(image)

        image = self.normalize(image)
        image = np.transpose(image, axes=[2, 0, 1])
        image = torch.FloatTensor(image)

        if self.test:
            return image, filename
        else:
            return image, label

    @staticmethod
    def normalize(image):
        std = np.std(image)
        if std > 0:
            image = (image - np.mean(image)) / std
        return image
