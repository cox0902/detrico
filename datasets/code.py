from typing import *
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.v2 as T
from util.box_ops import box_xyxy_to_cxcywh


class ImageCodeDataset(Dataset):

    def __init__(self,
                 image_path: str,
                 code_path: str,
                 split):
        super().__init__()
        self.transform = T.Compose([
            T.ToDtype(torch.float, scale=True),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            T.ToPureTensor()
        ])
        self.debug = []

        self.num_classes = 82

        self.hi = h5py.File(image_path, "r")
        self.images = self.hi["images"]
        self.labels = self.hi["labels"]   
        self.rects = self.hi["rects"]

        self.hc = h5py.File(code_path, "r")
        self.idx = self.hc["idx"]
        self.pid = self.hc["pid"]
        self.piv = self.hc["piv"]

        if split is not None:
            self.split = np.unique(self.idx[split])
            # print(self.split)
        else:
            self.split = np.unique(self.idx[:])
        assert len(self.split) <= len(self.images), (len(self.split), len(self.images))
    
    def __len__(self) -> int:
        if self.split is not None:
            return len(self.split)
        return len(self.images) 
    
    def __idx(self, i: int) -> int:
        if self.split is not None:
            return self.split[i]
        return i
    
    def __getitem__(self, index: int) -> Dict:
        w, h = 256, 256

        img_idx = self.__idx(index)
        # image = torch.from_numpy(self.images[img_idx])
        image = self.images[img_idx]
        if self.transform is not None:
            image = torch.from_numpy(image)
            image = self.transform(image)

        boxes, labels = [], []

        indices = np.where(self.idx[:] == img_idx)[0]
        # print(indices)
        for i in indices:
            pid = self.pid[i]
            piv = self.piv[i]
            assert pid != -1

            logical = ((self.labels[:, 0] == img_idx) & (self.labels[:, 1] == pid))

            loc = np.where(logical)
            if len(loc[0]) == 0:
                continue

            assert len(loc[0]) == 1
            rect = self.rects[loc[0]][0]
            boxes.append(rect)

            category_id = piv - 8
            labels.append(category_id)

        boxes = torch.as_tensor(np.array(boxes), dtype=torch.float32)
        boxes = box_xyxy_to_cxcywh(boxes)
        boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)

        return image, {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.as_tensor([img_idx]),
            "orig_size": torch.as_tensor([w, h]),
            "size": torch.as_tensor([w, h])
        }


def build(image_set, args):
    if image_set == "val":
        image_set = "valid"
    split = np.load(args.split_path)
    dataset = ImageCodeDataset(args.image_path, args.code_path, split[image_set])
    return dataset
