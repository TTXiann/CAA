import os
from os.path import join as osp
from PIL import Image
from torch.utils.data import Dataset as TorchDataset


# Short alias -> dataset folder under {data_folder}/RSICD
DATASET_FOLDERS = {
    'levir': 'LEVIR-CD256',
    'levir+': 'LEVIR-CD+256',
    'whu': 'WHU-CD256',
    'sysu': 'SYSU-CD256',
    'dsifn': 'DSIFN',
    'clcd': 'CLCD',
}


def resolve_dataset_folder(dataset):
    """Map a short alias (case insensitive) or a folder name to the folder name."""
    key = str(dataset).lower()
    if key in DATASET_FOLDERS:
        return DATASET_FOLDERS[key]
    for folder in DATASET_FOLDERS.values():
        if key == folder.lower():
            return folder
    raise ValueError(f'{dataset} Dataset not supported')


def canonical_dataset_alias(dataset):
    """Return the lowercase short alias used for experiment folders and logging."""
    folder = resolve_dataset_folder(dataset)
    return next(k for k, v in DATASET_FOLDERS.items() if v == folder)


def txt_to_list(file_path, strip_lines=True, skip_empty=True):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        if strip_lines:
            lines = [line.strip() for line in lines]
        if skip_empty:
            lines = [line for line in lines if line]

        return sorted(lines)
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: file '{file_path}' not found")
    except Exception as e:
        raise Exception(f"Error: cannot read file: {e}")


def get_image_paths(dataset_folder, mode):
    """Read {mode}.txt under list/ and build A/, B/ and label/ paths."""
    image_names = txt_to_list(os.path.join(dataset_folder, 'list', f'{mode}.txt'))
    before_image_paths, after_image_paths, label_paths = [], [], []
    for name in image_names:
        before_image_paths.append(os.path.join(dataset_folder, 'A', name))
        after_image_paths.append(os.path.join(dataset_folder, 'B', name))
        label_paths.append(os.path.join(dataset_folder, 'label', name))

    assert len(before_image_paths) == len(after_image_paths) == len(label_paths), \
        "A/B/label counts differ; check the list file and folders"

    return before_image_paths, after_image_paths, label_paths


class Dataset(TorchDataset):
    def __init__(self, dataset, mode, transform=None, root=None):
        if root is None:
            root = os.environ.get('CD_DATA_ROOT', '/project/datasets')
        self.root = root
        self.dataset = resolve_dataset_folder(dataset)

        dataset_folder = osp(root, 'RSICD', self.dataset)

        (self.before_image_paths, self.after_image_paths, self.label_paths) = get_image_paths(dataset_folder, mode)

        self.transform = transform

        print(f"[{dataset}][{mode}] {len(self.before_image_paths)} samples")

    def __len__(self):
        return len(self.before_image_paths)

    def __getitem__(self, idx):
        before_image_name = self.before_image_paths[idx]
        after_image_name = self.after_image_paths[idx]
        label_name = self.label_paths[idx]

        image1 = Image.open(before_image_name).convert('RGB')
        image2 = Image.open(after_image_name).convert('RGB')
        label  = Image.open(label_name).convert('L')

        if self.transform is not None:
            image1, image2, label = self.transform(image1, image2, label)

        return image1, image2, label, label_name
