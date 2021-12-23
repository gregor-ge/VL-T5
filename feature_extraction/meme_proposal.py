# coding=utf-8
import detectron2
from detectron2.data import MetadataCatalog

from detectron2_proposal_maxnms import collate_fn, NUM_OBJECTS, DIM, build_model, doit
from torch.utils.data import Dataset, DataLoader
import cv2
from pathlib import Path
import argparse
import os
import numpy as np
import torch
import h5py
from torchvision.ops import nms
from tqdm import tqdm

from detectron2.utils.visualizer import Visualizer


# Load VG Classes
data_path = 'demo/data/genome/1600-400-20'
D2_ROOT = os.path.dirname(os.path.dirname(detectron2.__file__))  # Root of detectron2
vg_classes = []
with open(os.path.join(data_path, 'objects_vocab.txt')) as f:
    for object in f.readlines():
        vg_classes.append(object.split(',')[0].lower().strip())

vg_attrs = []
with open(os.path.join(data_path, 'attributes_vocab.txt')) as f:
    for object in f.readlines():
        vg_attrs.append(object.split(',')[0].lower().strip())
MetadataCatalog.get("vg").thing_classes = vg_classes
MetadataCatalog.get("vg").attr_classes = vg_attrs


class COCODataset(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.image_path_list = list(tqdm(image_dir.iterdir()))
        self.n_images = len(self.image_path_list)

        # self.transform = image_transform

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        image_path = self.image_path_list[idx]
        image_id = image_path.stem

        img = cv2.imread(str(image_path))

        return {
            'img_id': image_id,
            'img': img
        }



def extract(output_fname, dataloader, desc, output_folder):
    detector = build_model()

    with h5py.File(output_fname, 'w') as f:
        with torch.no_grad():
            for i, batch in tqdm(enumerate(dataloader),
                                 desc=desc,
                                 ncols=150,
                                 total=len(dataloader)):

                img_ids = batch['img_ids']
                # feat_list, info_list = feature_extractor.get_detectron_features(batch)

                imgs = batch['imgs']

                assert len(imgs) == 1

                img = imgs[0]
                img_id = img_ids[0]

                try:
                    instances, features = doit(img, detector)

                    instances = instances.to('cpu')
                    features = features.to('cpu')

                    num_objects = len(instances)

                    assert num_objects == NUM_OBJECTS, (num_objects, img_id)
                    assert features.shape == (NUM_OBJECTS, DIM)

                    grp = f.create_group(img_id)
                    grp['features'] = features.numpy()  # [num_features, 2048]
                    grp['obj_id'] = instances.pred_classes.numpy()
                    grp['obj_conf'] = instances.scores.numpy()
                    grp['attr_id'] = instances.attr_classes.numpy()
                    grp['attr_conf'] = instances.attr_scores.numpy()
                    grp['boxes'] = instances.pred_boxes.tensor.numpy()
                    grp['img_w'] = img.shape[1]
                    grp['img_h'] = img.shape[0]

                    instances.remove("attr_scores")
                    instances.remove("attr_classes")

                    v = Visualizer(img[:, :, :], MetadataCatalog.get("vg"), scale=1.2)
                    v = v.draw_instance_predictions(instances)
                    v.save(str(output_folder.joinpath(f"roi_{img_id}.jpg")))

                except Exception as e:
                    print(batch)
                    print(e)
                    continue



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', default=1, type=int, help='batch_size')
    parser.add_argument('--cocoroot', type=str, default=r'D:\Research\projects\misrik\4chan\roi_examples')
    parser.add_argument('--split', type=str, default='valid', choices=['train', 'valid', 'test'])

    args = parser.parse_args()

    SPLIT2DIR = {
        'train': 'train2014',
        'valid': 'val2014',
        'test': 'test2015',
    }

    coco_dir = Path(args.cocoroot).resolve()

    dataset_name = 'COCO'

    out_dir = coco_dir.joinpath("roi")
    if not out_dir.exists():
        out_dir.mkdir()
    coco_img_dir = coco_dir.joinpath("img")
    print('Load images from', coco_img_dir)
    print('# Images:', len(list(coco_img_dir.iterdir())))

    dataset = COCODataset(coco_img_dir)

    dataloader = DataLoader(dataset, batch_size=args.batchsize,
                            shuffle=False, collate_fn=collate_fn, num_workers=4)

    output_fname = out_dir.joinpath(f'{args.split}_boxes{NUM_OBJECTS}.h5')
    print('features will be saved at', output_fname)

    desc = f'{dataset_name}_{args.split}_{(NUM_OBJECTS, DIM)}'

    extract(output_fname, dataloader, desc, out_dir)
