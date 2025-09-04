from typing import *
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.v2 as T
from util.box_ops import box_xyxy_to_cxcywh
from util.tree import TreeNode


class ImageCodeDataset(Dataset):

    def __init__(self,
                 image_path: str,
                 code_path: str,
                 split,
                 mask_rate: float = 0):
        super().__init__()
        self.transform = T.Compose([
            T.ToDtype(torch.float, scale=True),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            T.ToPureTensor()
        ])
        self.debug = []

        self.mask_rate = mask_rate
        self.num_classes = 1

        self.hi = h5py.File(image_path, "r")
        self.images = self.hi["images"]
        self.labels = self.hi["labels"]   
        self.rects = self.hi["rects"]

        self.hc = h5py.File(code_path, "r")
        self.max_len = self.hc.attrs["max_len"]
        self.codes = self.hc["ivs"]
        self.code_lens = self.hc["les"]
        self.ids = self.hc["ids"]

        self.split = split
    
    def __len__(self) -> int:
        if self.split is not None:
            return len(self.split)
        return len(self.codes) 
    
    def __idx(self, i: int) -> int:
        if self.split is not None:
            return self.split[i]
        return i
    
    def __get_rect(self, image_idx, code_idx):
        loc = np.where(np.logical_and(
            self.labels[:, 0] == image_idx,
            self.labels[:, 1] == code_idx
        ))
        assert len(loc[0]) == 1
        return self.rects[loc[0]]
    
    def __getitem__(self, index: int) -> Dict:
        w, h = 256, 256

        img_idx = code_idx = self.__idx(index)
        # image = torch.from_numpy(self.images[img_idx])
        image = self.images[img_idx]

        # code = self.codes[code_idx]

        ids = self.ids[code_idx]
        ivs = self.codes[code_idx]

        count = len(ivs[ivs > 7])

        if self.mask_rate > 0 and np.random.rand() < self.mask_rate and count > 2:
            # print("src:" + "".join([f"{each:4}" for each in ivs[ivs > 0]]))
            tree = TreeNode.build_tree(ivs, ids)
            nodes = tree.ravel()
            nodes = np.random.choice(nodes[2:], 1, replace=False)
            mask = nodes[0].mask
            rect = self.__get_rect(img_idx, mask)
            x0, y0, x1, y1 = rect[0]
            x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
            x0 = min(max(0, x0), 255)
            y0 = min(max(0, y0), 255)
            x1 = min(max(0, x1), 255)
            y1 = min(max(0, y1), 255)
            # print(rect)
            # print(image)
            image[:, y0:y1, x0:x1] = 255
            nodes[0].delete_sub()
            ivs, ids = tree.build_list_with_mask()
            # print("dst:" + "".join([f"{each:4}" for each in ivs]))
            ivs, ids = np.array(ivs), np.array(ids)

        rects = np.stack((np.zeros_like(ivs, dtype=np.float32), ) * 4, axis=-1)
        for i, (each_id, each_iv) in enumerate(zip(ids, ivs)):
            if each_iv <= 7:
                continue
            rects[i] = self.__get_rect(img_idx, each_id)

        boxes = rects[ivs > 7]
        if self.num_classes == 1:
            labels = np.zeros_like(ivs[ivs > 7])
        else:
            labels = ivs[ivs > 7] - 8

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        boxes = box_xyxy_to_cxcywh(boxes)
        boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)

        if self.transform is not None:
            image = torch.from_numpy(image)
            image = self.transform(image)
        
        code = torch.zeros((307, ), dtype=torch.int64)
        code[:len(ivs)] = ivs

        return image, {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.as_tensor([img_idx], dtype=torch.int64),
            "orig_size": torch.as_tensor([w, h]),
            "size": torch.as_tensor([w, h]),
            "code": code
        }


def build(image_set, args):
    if image_set == "val":
        image_set = "valid"
    split = np.load(args.split_path)
    mask_rate = 0 if image_set != "train" else args.mask_rate
    dataset = ImageCodeDataset(args.image_path, args.code_path, split[image_set],
                               mask_rate=mask_rate)
    return dataset
