import json
import os

import torch
from PIL import Image
from torch.utils.data import Dataset

from dataset.cc import transforms

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

# Short alias -> annotation tag, image folder and split mapping
_CC_DATASETS = {
    'levir': {
        'tag': 'LEVIR_CC_5_cap_per_img_5_min_word_freq',
        'images': 'Levir-CC-dataset/images',
        'annotations': 'levir-cc',
        'image_dirs': {'TRAIN': 'train', 'VAL': 'val', 'TEST': 'test'},
        'id_files': {'TRAIN': 'train', 'VAL': 'val', 'TEST': 'test'},
        'A': 'A',
        'B': 'B',
    },
    'dubai': {
        'tag': 'DUBAI',
        'images': 'DUBAI',
        'annotations': 'dubai-cc',
        'image_dirs': {'TRAIN': '', 'VAL': '', 'TEST': ''},
        'id_files': {'TRAIN': 'train', 'VAL': 'test', 'TEST': 'test'},
        'A': '500_2000',
        'B': '500_2010',
    },
}


def cc_dataset_key(dataset):
    """Accept a short alias (levir/dubai) or a legacy annotation tag."""
    name = str(dataset)
    if name.lower() in _CC_DATASETS:
        return name.lower()
    upper = name.upper()
    if 'LEVIR' in upper:
        return 'levir'
    if 'DUBAI' in upper:
        return 'dubai'
    raise ValueError(f'unsupported CC dataset: {dataset}')


def cc_dataset_spec(dataset):
    return _CC_DATASETS[cc_dataset_key(dataset)]


def cc_annotation_dir(data_folder, dataset):
    """Resolve annotations under {data_folder}/RSICC, with the repo data/ as fallback."""
    spec = cc_dataset_spec(dataset)
    candidates = [
        os.path.join(data_folder, 'RSICC', 'annotations'),
        os.path.join(data_folder, 'RSICC'),
        os.path.join(_REPO_ROOT, 'data', spec['annotations']),
    ]
    for folder in candidates:
        if os.path.isfile(os.path.join(folder, f'WORDMAP_{spec["tag"]}.json')):
            return folder
    raise FileNotFoundError(
        f'CC annotations for {dataset} not found, tried:\n  ' + '\n  '.join(candidates))


def cc_wordmap_path(data_folder, dataset):
    return os.path.join(cc_annotation_dir(data_folder, dataset),
                        f'WORDMAP_{cc_dataset_spec(dataset)["tag"]}.json')


class CaptionDataset(Dataset):
    def __init__(self, args, split):
        assert split in {'TRAIN', 'VAL', 'TEST'}
        self.split = split
        spec = cc_dataset_spec(args.dataset)
        ann_dir = cc_annotation_dir(args.data_folder, args.dataset)
        tag = spec['tag']

        with open(os.path.join(ann_dir, f'{split}_CAPTIONS_{tag}.json'), 'r') as f:
            self.captions = torch.LongTensor(json.load(f))[:, :args.max_len]

        self.img_path = os.path.join(
            args.data_folder, 'RSICC', spec['images'], spec['image_dirs'][split])
        self.pathA = spec['A']
        self.pathB = spec['B']

        with open(os.path.join(ann_dir, f'{spec["id_files"][split]}_ids.json'), 'r') as f:
            self.ids = json.load(f)

        # train: one caption per sample; eval: 5 refs per image
        self.cpi = 5 if split == 'TRAIN' else 1
        self.dataset_size = len(self.captions) if split == 'TRAIN' else len(self.captions) // 5

        # substring match on --backbone, first hit wins
        backbone = next((b for b in ('clip', 'mae', 'dino') if b in args.backbone), None)
        if backbone == 'dino':
            self.transform = transforms.DINOTransform()
        elif backbone in ('clip', 'mae'):
            self.transform = transforms.CLIPMAETransform(backbone)
        else:
            raise ValueError(f'{args.backbone} Model not supported')

    def __getitem__(self, idx):
        ids = self.ids[idx // self.cpi]

        img1 = Image.open(os.path.join(self.img_path, self.pathA, ids)).convert('RGB')
        img2 = Image.open(os.path.join(self.img_path, self.pathB, ids)).convert('RGB')
        img1, img2 = self.transform(img1, img2)

        if self.split == 'TRAIN':
            return img1, img2, self.captions[idx]
        return img1, img2, self.captions[idx * 5:(idx + 1) * 5]

    def __len__(self):
        return self.dataset_size
