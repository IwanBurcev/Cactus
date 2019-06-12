import cv2
import numpy as np
import pandas as pd
from pathlib import Path


def data_split(train_folder, anno_file):

    def read_samples(df):
        samples = []
        for n, image_info in df.iterrows():
            image_id = image_info.id
            has_cactus = image_info.has_cactus
            img = cv2.imread(str(train_folder.joinpath(image_id)))
            samples.append((img, has_cactus))

        return samples

    df = pd.read_csv(anno_file)

    np.random.seed(1)
    train_mask = np.random.rand(len(df)) < 0.8

    train_df = df[train_mask]  # 10534=1 3476=0
    val_df = df[~train_mask]  # 2602=1 888=0

    train_samples = read_samples(train_df)
    val_samples = read_samples(val_df)


    return train_samples, val_samples



data_root = Path('/media/eclipser/storage/kaggle/cactus/data/')
train_data = data_root.joinpath('train')
anno_file = data_root.joinpath('train.csv')

train_data, val_data = data_split(train_data, anno_file)