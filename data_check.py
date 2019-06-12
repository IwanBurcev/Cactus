import cv2
import pandas as pd
from pathlib import Path

data_root = Path('/media/eclipser/storage/kaggle/cactus/data/')

train_data = data_root.joinpath('train')
anno_file = data_root.joinpath('train.csv')   # 4363 has_cactus == 0 13136 has_cactus == 1

annotation = pd.read_csv(anno_file)

for file in train_data.glob('*.jpg'):
    has_cactus = annotation[annotation.id == file.name].has_cactus.values[0]
    img = cv2.imread(str(file))
    cv2.imshow(f'cactus {bool(has_cactus)}', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


