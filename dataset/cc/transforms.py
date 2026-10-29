import torch
import torchvision.transforms as transforms
from torchvision.transforms import v2


###### CLIP / MAE

class CLIPMAETransform:
    """Joint transform for CLIP/MAE: normalisation only, no augmentation."""

    def __init__(self, mode):
        if mode == 'clip':
            MEAN = (0.48145466, 0.4578275, 0.40821073)
            STD = (0.26862954, 0.26130258, 0.27577711)
        elif mode == 'mae':
            MEAN = (0.485, 0.456, 0.406)
            STD = (0.229, 0.224, 0.225)
        else:
            raise ValueError("mode is not supported")

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=MEAN, std=STD)

    def __call__(self, image1, image2):
        image1 = self.to_tensor(image1)
        image2 = self.to_tensor(image2)

        image1 = self.normalize(image1)
        image2 = self.normalize(image2)
        return image1, image2


###### DINO

class DINOTransform:
    """Joint transform for the DINO family (ImageNet mean/std, normalisation only, no augmentation)."""

    def __init__(self):
        self.to_tensor = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ])
        self.normalize = v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    def __call__(self, image1, image2):
        image1 = self.normalize(self.to_tensor(image1))
        image2 = self.normalize(self.to_tensor(image2))
        return image1, image2
