import os
import numpy as np
from shutil import copyfile

base_dir = '/Users/canrobins13/Desktop/230/'
data_dir = base_dir + 'og_data/'
dest_base_dir = base_dir + 'data/'
train_dir = data_dir + 'train/'
val_dir = data_dir + 'val/'

for split in ['train', 'val', 'test']:
    for data_type in ['images', 'labelTxt']:
        dir_path = os.path.join(dest_base_dir, split, data_type)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

og_train_labels = os.listdir(os.path.join(data_dir, 'train/labelTxt'))
og_val_labels = os.listdir(os.path.join(data_dir, 'val/labelTxt'))

all_labels = og_train_labels + og_val_labels

all_basenames = [os.path.splitext(label_name)[0] for label_name in all_labels]
np.random.shuffle(all_basenames)
n = len(all_basenames)
train_basenames = all_basenames[:int(0.6*n)]
val_basenames = all_basenames[int(0.6*n):int(0.8*n)]
test_basenames = all_basenames[int(0.8*n):]


def copy_files(basenames, dest_dir):
    for basename in basenames:
        if os.path.exists(os.path.join(train_dir, 'labelTxt', basename) + '.txt'):
            base_dir = train_dir
        else:
            base_dir = val_dir
        img_src = os.path.join(base_dir, 'images', basename) + '.png'
        label_src = os.path.join(base_dir, 'labelTxt', basename) + '.txt'
        img_dest = os.path.join(dest_dir, 'images', basename) + '.png'
        label_dest = os.path.join(dest_dir, 'labelTxt', basename) + '.txt'
        copyfile(img_src, img_dest)
        copyfile(label_src, label_dest)


copy_files(train_basenames, os.path.join(dest_base_dir, 'train'))
copy_files(val_basenames, os.path.join(dest_base_dir, 'val'))
copy_files(test_basenames, os.path.join(dest_base_dir, 'test'))
