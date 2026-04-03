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


class CachedLatentDataset(datasets.DatasetFolder):
    """Loads cached VAE moments (.npz, from main_cache.py) along with
    original images.

    Compatible with the cache format produced by ``main_cache.py``:

        cached_path/
          class_a/
            img001.JPEG.npz   -> {'moments': ndarray, 'moments_flip': ndarray}
            ...
          class_b/
            ...

    Labels are inferred from the folder structure (same as ImageFolder).
    Original images are loaded from ``data_path`` using the same relative
    path (minus the ``.npz`` suffix).

    At each access the dataset:
      1. Randomly picks original or flipped moments (hflip augmentation).
      2. Samples ``z`` from the diagonal‑Gaussian posterior encoded in the
         moments, then normalises: ``(z - latent_mean) / latent_std``.
      3. Loads the corresponding source image (with matching flip) so that
         REPA or other pixel‑level losses remain usable.

    Args:
        root: Directory of cached ``.npz`` files (same structure as an
              ImageFolder, one sub‑dir per class).
        data_path: Root of the original image dataset.  The relative path
                   of each ``.npz`` file (with the ``.npz`` suffix removed)
                   is joined with ``data_path`` to locate the source image.
        transform: Transform for the source image.  Must **not** include
                   ``RandomHorizontalFlip`` — flipping is handled internally
                   to keep latent and image consistent.
        latent_mean: Mean used for latent normalisation (default 0).
        latent_std: Std used for latent normalisation (default 1).
    """

    def __init__(self, root: str, data_path: str, transform=None,
                 latent_mean: float = 0.0, latent_std: float = 1.0):
        super().__init__(root, loader=None, extensions=(".npz",))
        from PIL import Image
        self.Image = Image
        self.data_path = data_path
        self.img_transform = transform
        self.latent_mean = latent_mean
        self.latent_std = latent_std

    def __getitem__(self, index: int):
        path, target = self.samples[index]

        # --- cached moments ---
        data = np.load(path)
        flip = torch.rand(1).item() < 0.5
        moments = torch.from_numpy(
            data['moments_flip'] if flip else data['moments']
        ).float()

        # Sample z from diagonal Gaussian, then normalise
        mean, logvar = moments.chunk(2, dim=0)  # [C, H, W] each
        logvar = logvar.clamp(-30.0, 20.0)
        std = torch.exp(0.5 * logvar)
        z = mean + std * torch.randn_like(mean)
        z = (z - self.latent_mean) / self.latent_std

        # --- original image ---
        rel = os.path.relpath(path, self.root)          # class/img.JPEG.npz
        img_rel = rel[: -len('.npz')]                    # class/img.JPEG
        img = self.Image.open(
            os.path.join(self.data_path, img_rel)
        ).convert('RGB')
        if self.img_transform is not None:
            img = self.img_transform(img)
        if flip:
            img = img.flip(dims=[-1])

        return z, target, img
