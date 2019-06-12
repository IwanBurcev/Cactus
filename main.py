import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader

from dataset import Dataset
from model import Model


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
    train_mask = np.random.rand(len(df)) < 0.80

    train_df = df[train_mask]  # 10534=1 3476=0
    val_df = df[~train_mask]  # 2602=1 888=0

    train_samples = read_samples(train_df)
    valid_samples = read_samples(val_df)

    return train_samples, valid_samples


data_root = Path('/media/eclipser/storage/kaggle/cactus/data/')
train_data = data_root.joinpath('train')
anno_file = data_root.joinpath('train.csv')

train_data, valid_data = data_split(train_data, anno_file)

train_dataloader = DataLoader(Dataset(train_data, training=True), batch_size=500)
valid_dataloader = DataLoader(Dataset(valid_data, training=False), batch_size=3500)

model = Model()

for epoch in range(0, 200):
    print(epoch)
    for data in train_dataloader:
        train_loss = model.train(data)

    for data in valid_dataloader:
        val_loss, accuracy = model.valid(data)
    print(accuracy)

test_data = pd.read_csv(data_root.joinpath('sample_submission.csv'))
test_dir = data_root.joinpath('test')
df = test_data.copy()

model.net.eval()

test_samples = []

for n, image_info in test_data.iterrows():
    img_path = str(test_dir.joinpath(image_info.id))
    img = cv2.imread(img_path)
    test_samples.append((img, img_path))

test_dataloader = DataLoader(Dataset(test_samples, training=False, test=True), batch_size=1)
output_csv = open('sample_submission.csv', 'w')
output_csv.write('id,has_cactus\n')

for data in test_dataloader:
    prediction = model.predict(data)
    img_id = Path(data[1][0]).name
    output_csv.write(f'{img_id},{prediction[0]}\n')

#    img = cv2.imread(data[1][0])
#    cv2.imshow(f'cactus {bool(prediction[0])}', img)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
