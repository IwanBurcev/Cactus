import numpy as np


class Dataset():
    def __init__(self, samples, training=False):
        self.samples = samples
        self.training = training

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        image, label = self.samples[item]

        image = self.normalize(image)

        return image, label

    @staticmethod
    def normalize(image):

        std = np.std(image)

        if std > 0:
            image = (image - np.mean(image)) / std

        return image
