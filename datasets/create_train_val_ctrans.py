"""create virus dataset for training and validation"""

import numpy as np
import pandas as pd
import mrcfile
import cv2
import glob

import os
import csv
import json
from copy import deepcopy
from denoise import denoise, denoise_jpg_image


output_dir = 'train_val_viruses/annotations/'
mrc_file_path = 'train_val_viruses/images/'
coord_file_path  = 'train_val_viruses/annotations/'

def json_worker(dataset,mrc_file_path,file_name,coord_file_path,image_id, anno_id):
    image_path = mrc_file_path+file_name
    print(image_path)
    with mrcfile.open(image_path, mode='r+', permissive=True) as mrc:
        read_image = deepcopy(mrc.data)

    if read_image is None:
        print("skipped file:\n {}".format(image_path))
        return dataset, anno_id

    
    read_image = denoise(read_image)
    print(read_image)
    print(read_image.shape)
    data_out = cv2.imwrite(mrc_file_path+file_name[:-4] + '.png', read_image)
    IMG_HEIGHT, IMG_WIDTH = read_image.shape
    #print('image height: {}'.format(IMG_HEIGHT))

    dataset['images'].append({
                        'id': image_id,
                        'file_name': file_name[:-4] + '.png',
                        'width': IMG_WIDTH,
                        'height': IMG_HEIGHT})

    #read particle coordinates
    particle_coord_path = os.path.join(coord_file_path, '') + file_name[:-4] + '.box'
    if not os.path.exists(particle_coord_path):
        print(f"coordinates not avaibale for {file_name} ----> Skipping Image ID : {image_id}")
        dataset['annotations'].append({
            'id': anno_id,  #annotation id of its own
            'category_id': 1,  # particle class
            'iscrowd': 0,
            'area': 0,
            'image_id': image_id,
            'bbox': [0,0,0,0],
            'segmentation': []
        })
        anno_id += 1
    
    else:
        boxes = pd.read_csv(particle_coord_path, sep='\t',header=None)
        boxes = boxes.values
        for i in range(len(boxes)):
            BOX_WIDTH = int(boxes[i,2])
            dataset['annotations'].append({
                'id': anno_id,  #annotation id of its own
                'category_id': 1,  # particle class
                'iscrowd': 1,
                'area': BOX_WIDTH * BOX_WIDTH,
                'image_id': image_id,
                'bbox': [int(k) for  k in boxes[i,:]],
                'segmentation': []
            })
            anno_id += 1
    return dataset, anno_id


def create_json_dataset(mrc_file_path, coord_file_path, output_dir):
    train_dataset = {'info': [], 'categories': [] , 'images': [],  'annotations': []}
    val_dataset = {'info': [], 'categories': [] , 'images': [],  'annotations': []}
    classes = ['particle']
    csv_file_not_count = 0
    val_anno_ids = 1
    train_anno_ids = 1



    file_names = [f.split('/')[-1] for f in sorted(glob.glob(mrc_file_path+'*.mrc'))]
    train_ids = np.random.choice(len(file_names), size = int(len(file_names)*0.7))
    for image_id, file_name in enumerate(file_names):
        # write image id, name, width and height
        if image_id in train_ids:
            train_dataset, train_anno_ids=json_worker(
                train_dataset,mrc_file_path,file_name,coord_file_path,
                image_id,train_anno_ids)
        else:
            val_dataset, val_anno_ids=json_worker(
                val_dataset,mrc_file_path,file_name,coord_file_path,
                image_id,val_anno_ids)

    print(val_dataset)



    print("------------------------  STATS   ---------------------")
    print(f"Total Micrographs : {len(file_names)}")
    print("------------------------     ---------------------")

    # save json annotation results
    val_name = os.path.join(output_dir, 'instances_val.json')
    with open(val_name, 'w') as f:
        json.dump(val_dataset, f)
    
    train_name = os.path.join(output_dir, 'instances_train.json')
    with open(train_name, 'w') as f:
        json.dump(train_dataset, f)

create_json_dataset(mrc_file_path, coord_file_path, output_dir)
