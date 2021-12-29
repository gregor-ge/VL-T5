import argparse
import csv
import json
import os
from collections import defaultdict

import clip
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class ImageDataset(Dataset):
    def __init__(self, image_dir, image_files, labels):
        self.image_dir = image_dir
        self.image_files = image_files
        self.labels = labels
        self.n_images = len(self.image_files)

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        file = self.image_files[idx]
        label = self.labels[idx]
        return {
            "img": Image.open(os.path.join(self.image_dir, file)),
            "label": label
        }


def collate_fn(batch):
    out = defaultdict(list)
    for item in batch:
        for k, v in item.items():
            out[k].append(v)
    return out

class Evaluator:
    def __init__(self, clip_name, dataloader, args, label2query):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(clip_name, device=self.device, download_root=args.download_root)
        self.dataloader = dataloader
        self.args = args
        self.label2query = label2query
        self.label2idx = {label: i for i, label in enumerate(label2query.keys())}
        self.idx2label = {v: k for k, v in self.label2idx.items()}

    def evaluate(self):
        text_inputs = torch.cat([clip.tokenize(query) for query in self.label2query.values()]).to(self.device)

        gold_labels = []
        predicted_labels = []

        # Calculate features
        with torch.no_grad():
            text_features = self.model.encode_text(text_inputs)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            for batch in tqdm(self.dataloader, total=len(self.dataloader), desc="Predicting"):
                imgs = torch.stack([self.preprocess(img_path) for img_path in batch["img"]]).to(self.device)
                img_features = self.model.encode_image(imgs)
                img_features /= img_features.norm(dim=-1, keepdim=True)
                similarity = (100.0 * img_features @ text_features.T).softmax(dim=-1)
                gold_labels.extend([self.label2idx[label] for label in batch["label"]])
                predicted_labels.extend(torch.argmax(similarity, dim=-1).cpu().tolist())

        hit = defaultdict(lambda: 0)
        total = defaultdict(lambda: 0)
        correct = defaultdict(lambda: 0)
        labels = list(self.label2idx.keys())
        for pred, gold in zip(predicted_labels, gold_labels):
            for label in labels:
                if pred == self.label2idx[label]:
                    hit[label] += 1
                    if label == self.idx2label[gold]:
                        correct[label] += 1
            total[self.idx2label[gold]] += 1

        result = {"precision": {}, "recall": {}, "f1": {}}
        for label in labels:
            try:
                precision = correct[label] / hit[label]
            except ZeroDivisionError:
                precision = 0
            try:
                recall = correct[label] / total[label]
            except ZeroDivisionError:
                recall = 0
            try:
                f1 = 2 * precision * recall / (precision + recall)
            except ZeroDivisionError:
                f1 = 0
            result["precision"][label] = precision
            result["recall"][label] = recall
            result["f1"][label] = f1
        makro_f1 = sum(result["f1"][label] for label in labels) / len(labels)
        sum_total = sum(total.values())
        micro_f1 = sum(result["f1"][label] * total[label] / sum_total for label in labels)
        acc = sum(correct[label] / sum_total for label in labels)
        result["makro_f1"] = makro_f1
        result["micro_f1"] = micro_f1
        result["accuracy"] = acc
        return result



def get_evaluator(args):
    task = args.task
    if task == "harmemes":
        labels = []
        image_files = []
        args.data_path = os.path.join(args.data_path, "harmemes")
        img_path = os.path.join(args.data_path, "images")
        split_dir = os.path.join(args.data_path, "annotations")
        data_info_path = os.path.join(split_dir, f'{args.split}.jsonl')
        with open(data_info_path, encoding="utf-8") as f:
            for row in f.readlines():
                row = json.loads(row)
                labels.append(row["labels"][0])
                image_files.append(row["image"])
        label2query = {
            "not harmful": "A harmless meme",
            "somewhat harmful": "A somewhat harmful meme",
            "very harmful": "A very harmful meme"
        }
    elif task == "multioff":
        labels = []
        image_files = []
        args.data_path = os.path.join(args.data_path, "MultiOFF_Dataset")
        img_path = os.path.join(args.data_path, "Labelled Images")
        split_dir = os.path.join(args.data_path, "Split Dataset")
        data_info_path = os.path.join(split_dir, f'{args.split}_meme_dataset.csv')
        with open(data_info_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                labels.append(row["label"])
                image_files.append(row["image_name"])
        label2query = {
            "offensive": "An offensive meme", #"An offensive meme",
            "Non-offensiv": "A meme", #"A harmless meme"
        }

    dataset = ImageDataset(img_path, image_files, labels)
    dataloader = DataLoader(dataset, batch_size=args.batchsize, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)
    evaluator = Evaluator(args.clip, dataloader, args, label2query)
    return evaluator



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', default=32, type=int, help='batch_size')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--data_path', type=str, default=r'D:\Research\projects\misrik\VL-T5\datasets')
    parser.add_argument('--split', type=str, default='valid')
    parser.add_argument('--task', type=str, default='multioff')
    parser.add_argument("--clip", type=str, default="RN50x4")
    parser.add_argument('--download_root', type=str, default=None)
    args = parser.parse_args()

    evaluator = get_evaluator(args)
    print(evaluator.evaluate())