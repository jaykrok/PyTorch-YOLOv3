import os
import numpy as np
import dota_utils as util
from PIL import Image
import shutil

BASE_DIR = '/Users/canrobins13/Desktop/230/data/'
wordname_15 = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
               'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter']

## trans dota format to format YOLO(darknet) required
def dota2darknet(imgpath, txtpath, dstpath, extractclassname):
    """
    :param imgpath: the path of images
    :param txtpath: the path of txt in dota format
    :param dstpath: the path of txt in YOLO format
    :param extractclassname: the category you selected
    :return:
    """
    objects = util.parse_dota_poly(txtpath)
    img = Image.open(imgpath)
    img_w, img_h = img.size
    # print img_w,img_h
    with open(dstpath, 'w') as f_out:
        for obj in objects:
            poly = obj['poly']
            bbox = np.array(util.dots4ToRecC(poly, img_w, img_h))
            if (sum(bbox <= 0) + sum(bbox >= 1)) >= 1:
                print('skipping {} (case 1)'.format(txtpath))
                continue
            if (obj['name'] in extractclassname):
                id = extractclassname.index(obj['name'])
            else:
                print('skipping {} (case 2)'.format(txtpath))
                continue
            outline = str(id) + ' ' + ' '.join(list(map(str, bbox)))
            f_out.write(outline + '\n')


def convert_labels(src_dir, dest_dir, image_dir):
    for label_name in os.listdir(src_dir):
        src_label_path = os.path.join(src_dir, label_name)
        dest_path = os.path.join(dest_dir, label_name)
        image_path = os.path.join(image_dir, os.path.splitext(label_name)[0] + '.png')
        dota2darknet(image_path, src_label_path, dest_path, wordname_15)


def main():
    for split in ['train', 'val', 'test']:
        src_dir = os.path.join(BASE_DIR, split, 'labelTxt')
        dest_dir = os.path.join(BASE_DIR, split, 'processed_labels')
        image_dir = os.path.join(BASE_DIR, split, 'images')

        if  os.path.exists(dest_dir):
            shutil.rmtree(dest_dir)
        os.makedirs(dest_dir)

        convert_labels(src_dir, dest_dir, image_dir)


if __name__ == '__main__':
    main()
