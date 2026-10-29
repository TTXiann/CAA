import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as v2F



def binary_rgb_to_class(x: torch.Tensor) -> torch.Tensor:
    # x: [1,H,W], uint8
    c = x[0]

    # 0 -> 0, anything non-zero (non-black) -> 1
    y = (c != 0).to(torch.long)

    return y


###### CLIP / MAE

class CLIPMAETransform:
    """Joint transform for CLIP/MAE; train=True enables flips, rotation and temporal swap."""

    def __init__(self, mode, train=False):

        if mode == 'clip':
            MEAN = (0.48145466, 0.4578275, 0.40821073)
            STD = (0.26862954, 0.26130258, 0.27577711)
        elif mode == 'mae':
            MEAN = (0.485, 0.456, 0.406)
            STD = (0.229, 0.224, 0.225)
        else:
            raise ValueError("mode is not supported")

        self.train_mode = train
        self.to_tensor = transforms.ToTensor()
        self.label_transform = transforms.Compose([  # v1: PIL 'L' -> uint8 [1,H,W] -> class indices
            transforms.PILToTensor(),
            transforms.Lambda(binary_rgb_to_class),
        ])
        self.normalize = transforms.Normalize(mean=MEAN, std=STD)

    def __call__(self, image1, image2, label):

        image1 = self.to_tensor(image1)
        image2 = self.to_tensor(image2)
        label = self.label_transform(label).unsqueeze(0)  # [1,H,W]

        if self.train_mode:
            if torch.rand(()) < 0.5:
                image1 = F.hflip(image1)
                image2 = F.hflip(image2)
                label = F.hflip(label)

            if torch.rand(()) < 0.5:
                image1 = F.vflip(image1)
                image2 = F.vflip(image2)
                label = F.vflip(label)

            if torch.rand(()) < 0.5:
                k = int(torch.randint(1, 4, (1,)).item())
                image1 = torch.rot90(image1, k, dims=(-2, -1))
                image2 = torch.rot90(image2, k, dims=(-2, -1))
                label = torch.rot90(label, k, dims=(-2, -1))

            if torch.rand(()) < 0.5:
                image1, image2 = image2, image1

        image1 = self.normalize(image1)
        image2 = self.normalize(image2)
        return image1, image2, label


###### DINO

class DINOTransform:
    """Joint transform for DINO (ImageNet mean/std, labels as class indices); train adds augmentation."""

    def __init__(self, train=False):
        self.train_mode = train

        self.to_tensor = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ])
        self.label_transform = v2.Compose([  # v2: PIL 'L' -> uint8 [1,H,W] -> class indices
            v2.ToImage(),
            v2.ToDtype(torch.uint8, scale=False),
            v2.Lambda(binary_rgb_to_class),
        ])
        self.normalize = v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    def __call__(self, image1, image2, label):
        image1 = self.to_tensor(image1)
        image2 = self.to_tensor(image2)
        label = self.label_transform(label).unsqueeze(0)  # [1,H,W]

        if self.train_mode:
            if torch.rand(()) < 0.5:
                image1 = v2F.horizontal_flip(image1)
                image2 = v2F.horizontal_flip(image2)
                label = v2F.horizontal_flip(label)

            if torch.rand(()) < 0.5:
                image1 = v2F.vertical_flip(image1)
                image2 = v2F.vertical_flip(image2)
                label = v2F.vertical_flip(label)

            if torch.rand(()) < 0.5:
                k = int(torch.randint(1, 4, (1,)).item())
                image1 = torch.rot90(image1, k, dims=(-2, -1))
                image2 = torch.rot90(image2, k, dims=(-2, -1))
                label = torch.rot90(label, k, dims=(-2, -1))

            if torch.rand(()) < 0.5:
                image1, image2 = image2, image1

        image1 = self.normalize(image1)
        image2 = self.normalize(image2)
        return image1, image2, label
