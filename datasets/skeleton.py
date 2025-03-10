# ------------------------------------------------------------------------
# Blumnet
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import cv2
import os
import torch
import pandas as pd
import numpy as np
import PIL.Image as Image
from torch.utils.data import Dataset
from .decomposition import decompose_skeleton
import datasets.sktransforms as T


class SkDataset(Dataset):
    def __init__(self, fileNames, rootDir, transforms, base_size=512, npt=2, rule='overlap_10_0.6'):
        super(SkDataset, self).__init__()
        self.rootDir = rootDir
        self.frame = pd.read_csv(fileNames, dtype=str, delimiter=' ', header=None)
        self.transforms = transforms
        self.base_size = base_size
        self.npt = npt
        self.rule = rule
        self.curve_length = eval(rule.split('_')[1])
        self.dia_iters = 2

    def id2name(self, id):
        image_fpath = os.path.join(self.rootDir, self.frame.iloc[id, 0])
        ann_fpath = os.path.join(self.rootDir, self.frame.iloc[id, 1])
        return image_fpath, ann_fpath

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        image_fpath, ann_fpath = self.id2name(idx)
        areas_fpath = f"{ann_fpath[:-4]}_mask.png"
        img, skeleton, sample = self.get_target(
            image_fpath, ann_fpath, areas_fpath=areas_fpath, base_size=self.base_size,
            npt=self.npt, rule=self.rule, dil_iters=self.dia_iters)

        sample.update({'id': idx, 'image': img, 'skeleton': skeleton})
        sample = self.transforms(sample)

        branches = sample.pop('branches')
        sample['curves'] = torch.cat([b['curves'] for b in branches], dim=0)
        sample['cids'] = torch.cat([b['cids'] for b in branches], dim=0)
        sample['clabels'] = torch.cat([b['clabels'] for b in branches], dim=0)
        sample['key_pts'] = sample['key_pts'][:, None, :]
        sample['keypoint_directions'] = sample['keypoint_directions'][:, None, :]
        img = sample.pop('image')
        tgt = sample
        return img, tgt

    @staticmethod
    def get_target(image_fpath, ann_fpath, areas_fpath=None,
                   base_size=512, npt=2, rule='overlap_10_0.6', dil_iters=2):

        if (not os.path.isfile(image_fpath)) or (not os.path.isfile(ann_fpath)):
            raise ValueError(f"wrong image_fpath: {image_fpath}, OR ann_fpath: {ann_fpath}")

        image = Image.open(image_fpath).convert("RGB")
        skeleton = cv2.imread(ann_fpath, 0).astype(np.float32)
        sk_img = Image.fromarray(skeleton.astype(np.float32))
        w, h = origin_size = image.size
        sk_h, sk_w = skeleton.shape[:2]
        assert w == sk_w and h == sk_h, f"w, h: {w, h}, sk_h, sk_w: {sk_h, sk_w}"
        if areas_fpath is not None and os.path.isfile(areas_fpath):
            graph_areas = cv2.imread(areas_fpath, 0)
            graph_values = set(np.reshape(graph_areas, (-1,)).tolist()) - {0}
            sk_masks = [(graph_areas == v) * skeleton for v in graph_values]
        else:
            sk_masks = [skeleton]

        fx = fy = base_size * 1.0 / max(w, h)
        sk_masks = [cv2.resize(sk, dsize=None, fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)
                    for sk in sk_masks]

        target = {"orig_size": origin_size, }  # "file_name": os.path.basename(image_fpath),
        target.update(
            decompose_skeleton(sk_masks, rule=rule, npt=npt, dil_iters=dil_iters))

        return image, sk_img, target


class SkTestset(Dataset):
    def __init__(self, fileNames, rootDir, transforms, base_size=512, npt=2, rule='overlap_10_0.6'):
        super(SkTestset, self).__init__()
        self.rootDir = rootDir
        self.frame = pd.read_csv(fileNames, dtype=str, delimiter=' ', header=None)
        self.transforms = transforms
        self.base_size = base_size
        self.npt = npt
        self.rule = rule

    def id2name(self, id):
        image_fpath = os.path.join(self.rootDir, self.frame.iloc[id, 0])
        ann_fpath = os.path.join(self.rootDir, self.frame.iloc[id, 1])
        return image_fpath, ann_fpath

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        image_fpath, ann_fpath = self.id2name(idx)
        img = Image.open(image_fpath).convert("RGB")
        skeleton = Image.fromarray(
            cv2.imread(ann_fpath, 0).astype(np.float32))
        sample = {'id': idx, 'image': img, 'skeleton': skeleton, 'branches': []}
        sample['orig_size'] = img.size

        if self.transforms is not None:
            sample = self.transforms(sample)
        branches = sample.pop('branches')
        img = sample.pop('image')
        tgt = sample
        return img, tgt


class SkInferset(Dataset):
    def __init__(self, im_names, rootDir, transforms, base_size=512, npt=2, rule='overlap_10_0.6'):
        super(SkInferset, self).__init__()
        self.rootDir = rootDir
        self.im_names = im_names
        self.transforms = transforms
        self.base_size = base_size
        self.npt = npt
        self.rule = rule

    def id2name(self, id):
        image_fpath = os.path.join(self.rootDir, self.im_names[id])
        return image_fpath, image_fpath

    def __len__(self):
        return len(self.im_names)

    def __getitem__(self, idx):
        image_fpath, _ = self.id2name(idx)
        img = Image.open(image_fpath).convert("RGB")
        w, h = img.size
        skeleton = Image.fromarray(np.zeros((h, w), dtype=np.float32))
        sample = {'id': idx, 'image': img, 'skeleton': skeleton, 'branches': []}
        sample['orig_size'] = img.size
        if self.transforms is not None:
            sample = self.transforms(sample)
        branches = sample.pop('branches')
        img = sample.pop('image')
        tgt = sample
        return img, tgt


def make_skeleton_transforms(is_train, rotate):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    train_scales = [384, 416, 448, 480, 512, 576]
    test_scales = [512]
    trans1 = [T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.RandomRotate90(), T.RandomResize(train_scales)
              ]
    trans2 = [T.ColorJitter(), normalize]

    if rotate:
        train_transform = trans1 + [T.RandomRotateAny(),] + trans2
    else:
        train_transform = trans1 + trans2

    if is_train:
        return T.Compose(train_transform)
    else:
        return T.Compose([
            T.RandomResize(test_scales),
            normalize,
        ])


def build_dataset(config, is_train):
    if is_train:
        split_file = config["train_split"]
    else:
        split_file = config["test_split"]
    dataset = SkDataset(split_file, config["data_root"],
                        transforms=make_skeleton_transforms(is_train, config["random_rotate"]),
                        base_size=512, npt=config["points_per_path"], rule=config["rule"])
    return dataset


def visualize_target(target):
    canvas = (target["skeleton"].numpy()*255).astype(np.uint8)[0]
    canvas = np.stack([canvas]*3, axis=2)

    keypoints = target["key_pts"].numpy().squeeze(1)
    keypoints_labels = target["plabels"].numpy()
    keypoint_directions = target["keypoint_directions"].numpy().squeeze(1)

    vis_result = visualize_keypoints(canvas, keypoints, keypoints_labels, keypoint_directions)
    return vis_result


def visualize_keypoints(canvas, keypoints, labels, directions):
    h, w = canvas.shape[:2]
    keypoints[:, 0] = keypoints[:, 0] * w
    keypoints[:, 1] = keypoints[:, 1] * h
    keypoints = keypoints.astype(np.int32)
    for i in range(keypoints.shape[0]):
        keypoint = keypoints[i]
        label = labels[i]
        keypoint_direction = directions[i]
        if np.linalg.norm(keypoint_direction) < 0.5:
            label = -1  # unsigned junction
        keypoint_direction = (keypoint_direction * 20).astype(np.int32)
        if label == 1:  # end point
            cv2.circle(canvas, (keypoint[0], keypoint[1]), 5, (0, 255, 0), -1)
            cv2.arrowedLine(canvas, (keypoint[0], keypoint[1]), (keypoint[0] + keypoint_direction[0], keypoint[1] + keypoint_direction[1]), (0, 255, 0), 2)
        elif label == 0:  # junction point
            cv2.circle(canvas, (keypoint[0], keypoint[1]), 5, (0, 0, 255), -1)
            cv2.arrowedLine(canvas, (keypoint[0], keypoint[1]), (keypoint[0] + keypoint_direction[0], keypoint[1] + keypoint_direction[1]), (0, 0, 255), 2)
        elif label == -1:  # unsigned junction
            cv2.circle(canvas, (keypoint[0], keypoint[1]), 10, (0, 0, 255), -1)
    return canvas


def tensor_to_cv(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = tensor * std + mean
    # Convert to numpy and transpose from (C,H,W) to (H,W,C)
    tensor = tensor.numpy().transpose(1, 2, 0)
    # Scale to 0-255 range and convert to uint8
    tensor = (tensor * 255).clip(0, 255).astype(np.uint8)
    # Convert from RGB to BGR for OpenCV
    tensor = tensor[:, :, ::-1]
    return tensor


if __name__ == "__main__":
    split_file = "data/sk1491/test/test_pair.lst"
    root_dir = "data"
    dataset = SkDataset(split_file, root_dir,
                        transforms=make_skeleton_transforms(False, False),
                        base_size=512, npt=2, rule='overlap_10_0.6')
    data_length = len(dataset)
    vis_root = "/home/jimzhang/Projects/BlumNetGraph/tmp/vis_graph"
    from tqdm import tqdm
    for i in tqdm(range(data_length)):
        img, tgt = dataset[i]
        img_cv = tensor_to_cv(img)
        vis_target = visualize_target(tgt)
        vis_combined = np.concatenate([img_cv, vis_target], axis=1)
        cv2.imwrite(f"{vis_root}/target_{i}.png", vis_combined)