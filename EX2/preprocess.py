'''
    Exercise 1:
        convert the flower_dataset to ImageNet format.
'''

import os
import argparse
from typing import List, Tuple
from glob import glob
import numpy as np

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', '-o', type=str, default='./data', help='Path to preprocessed data')
    parser.add_argument('--dataset_path', '-d', type=str, default='./flower_dataset', help='Path to flower dataset')
    parser.add_argument('--seed', '-s', type=int, default=0, help='random seed')
    return parser

def read_all_data(dataset_path: str) -> List[str]:
    '''
        read all image, return a list of image path
    '''
    return sorted(glob('*/*.jpg', root_dir=dataset_path))

def train_val_split(img: List[str], seed: int) -> tuple[List[str], List[str]]:
    '''
        Uniformly sample training set and validation set from the flower_dataset.
    '''
    np.random.seed(seed)
    np.random.shuffle(img)

    train_len = int(np.round((len(img) * 0.8)))

    train_list = img[:train_len]
    val_list = img[train_len:]
    return train_list, val_list

def write_classes_txt(mapping: dict, output_path: str):
    '''
        write all names flower categories into file 'classes.txt' with each line representing one class.
    '''
    with open(os.path.join(output_path, 'classes.txt'), 'w+') as f:
        for key, classes in mapping.items():
            f.writelines(f'{key} {classes}' + '\n')

def write_annotation_lists(mapping: dict, train_list: List[str], val_list: List[str], output_path: str):
    '''
        Generate training and validation sets annotation lists: 'train.txt' and 'val.txt'.
    '''
    with open(os.path.join(output_path, 'train.txt'), 'w+') as f:
        for img in train_list:
            classes = mapping[img.split('/')[-2]]
            f.writelines(f'{img} {classes}' + '\n')

    with open(os.path.join(output_path, 'val.txt'), 'w+') as f:
        for img in val_list:
            classes = mapping[img.split('/')[-2]]
            f.writelines(f'{img} {classes}' + '\n')

def write_train_val_img(mapping: dict, train_list: List[str], val_list: List[str], dataset_path: str, output_path: str):
    '''
        copy image in training set to data/train/
    '''
    os.makedirs(os.path.join(output_path, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'val'), exist_ok=True)

    for key, classes in mapping.items():
        os.makedirs(os.path.join(output_path, 'train', key), exist_ok=True)
        os.makedirs(os.path.join(output_path, 'val', key), exist_ok=True)

    for img in train_list:
        output = os.path.join(output_path, 'train', img)
        os.system(f'cp {os.path.join(dataset_path, img)} {output}')

    for img in val_list:
        output = os.path.join(output_path, 'val', img)
        os.system(f'cp {os.path.join(dataset_path, img)} {output}')

def main():
    parser = get_args_parser()
    args = parser.parse_args()

    dataset_path = args.dataset_path
    output_path = args.output_path
    print(f'Preparing flower dataset: {dataset_path}')
    os.makedirs(output_path, exist_ok=True)

    img = read_all_data(dataset_path)
    train_list, val_list = train_val_split(img, args.seed)
    
    mapping = {'daisy': 0, 'dandelion': 1, 'rose': 2, 'sunflower': 3, 'tulip': 4}
    write_classes_txt(mapping, output_path)
    write_annotation_lists(mapping, train_list, val_list, output_path)
    write_train_val_img(mapping, train_list, val_list, dataset_path, output_path)


if __name__ == '__main__':
    main()