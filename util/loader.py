import os
import numpy as np

import torch
import torchvision.datasets as datasets


class ImageFolderWithFilename(datasets.ImageFolder):
    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target, filename).
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        filename = path.split(os.path.sep)[-2:]
        filename = os.path.join(*filename)
        return sample, target, filename


class CachedFolder(datasets.DatasetFolder):
    def __init__(
            self,
            root: str,
    ):
        super().__init__(
            root,
            loader=None,
            extensions=(".npz",),
        )

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (moments, target).
        """
        path, target = self.samples[index]

        data = np.load(path)
        if torch.rand(1) < 0.5:  # randomly hflip
            moments = data['moments']
        else:
            moments = data['moments_flip']

        return moments, target


class CachedLatentDataset(torch.utils.data.Dataset):
    """Loads pre-cached VAE latents (.pt files) along with original images.

    Expected directory structure:
        cached_path/
            metadata.json
            latents/
                0000000.pt  -> {'latent': tensor, 'latent_flip': tensor,
                                'label': int, 'img_relpath': str}
                ...

    Each .pt file stores the output of vae.encode() for both the original
    and horizontally-flipped image, plus the relative path to the source
    image so the original pixels can be loaded at training time (needed for
    REPA and visualization).

    Args:
        root: Path to cached latent directory.
        data_path: Root directory of the original image dataset.
                   Combined with img_relpath from each .pt file to load
                   the source image.
        transform: Transform applied to the original image (should NOT
                   include RandomHorizontalFlip — flip is handled internally
                   to stay consistent with the latent selection).
    """

    def __init__(self, root: str, data_path: str, transform=None):
        from PIL import Image
        self.Image = Image
        latent_dir = os.path.join(root, 'latents')
        self.files = sorted([
            os.path.join(latent_dir, f)
            for f in os.listdir(latent_dir)
            if f.endswith('.pt')
        ])
        if len(self.files) == 0:
            raise RuntimeError(f"No .pt files found in {latent_dir}")
        self.data_path = data_path
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index: int):
        data = torch.load(self.files[index], map_location='cpu', weights_only=True)
        flip = torch.rand(1).item() < 0.5
        latent = data['latent_flip'] if flip else data['latent']
        label = data['label']

        # Load and transform original image
        img_path = os.path.join(self.data_path, data['img_relpath'])
        img = self.Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if flip:
            img = img.flip(dims=[-1])

        return latent, label, img
