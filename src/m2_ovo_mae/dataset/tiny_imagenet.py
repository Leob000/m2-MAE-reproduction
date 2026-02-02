import shutil
import zipfile
from pathlib import Path

import requests
from torchvision.datasets import ImageFolder, VisionDataset
from tqdm import tqdm

URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
MD5 = "90528d7ca1a48142e341f4ef8d21d0de"


class TinyImageNet(VisionDataset):
    """Tiny ImageNet Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``tiny-imagenet-200`` exists or will be saved to if download is set to True.
        split (string, optional): The dataset split, supports ``train``, ``val``, or ``test``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        download: bool = False,
        transform=None,
        target_transform=None,
    ):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.dataset_path = Path(root) / "tiny-imagenet-200"
        self.split = split

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted. You can use download=True"
            )

        self._format_val_folders()

        # Define the directory for the chosen split
        split_dir = self.dataset_path / split

        self.data = ImageFolder(
            root=str(split_dir), transform=transform, target_transform=target_transform
        )
        self.classes = self.data.classes
        self.class_to_idx = self.data.class_to_idx

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def _check_integrity(self) -> bool:
        """Checks if the dataset files are present on disk."""
        return (self.dataset_path / "wnids.txt").exists() and (
            self.dataset_path / "train"
        ).exists()

    def download(self):
        """Downloads and extracts the Tiny ImageNet dataset."""
        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        self.dataset_path.parent.mkdir(parents=True, exist_ok=True)
        zip_path = self.dataset_path.parent / "tiny-imagenet-200.zip"

        if not zip_path.exists():
            print(f"Downloading {URL}...")
            response = requests.get(URL, stream=True)
            total_size_in_bytes = int(response.headers.get("content-length", 0))
            block_size = 1024
            progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)

            with open(zip_path, "wb") as file:
                for data in response.iter_content(block_size):
                    progress_bar.update(len(data))
                    file.write(data)
            progress_bar.close()

        print("Extracting...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(self.dataset_path.parent)

    def _format_val_folders(self):
        """Reorganizes the val directory into standard ImageFolder structure.

        TinyImageNet val set comes as a flat folder of images and a 'val_annotations.txt' file.
        We move images into subfolders based on their class ID found in annotations.
        """
        val_dir = self.dataset_path / "val"
        images_dir = val_dir / "images"
        annotations_file = val_dir / "val_annotations.txt"

        # If images_dir exists, we need to format it.
        if images_dir.exists():
            print("Formatting validation set...")
            with open(annotations_file) as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split("\t")
                filename = parts[0]
                class_id = parts[1]

                class_dir = val_dir / class_id
                class_dir.mkdir(exist_ok=True)

                src = images_dir / filename
                dst = class_dir / filename

                if src.exists():
                    shutil.move(str(src), str(dst))

            # Cleanup empty images folder
            if not any(images_dir.iterdir()):
                images_dir.rmdir()

        # Final integrity check: Ensure we have subfolders
        # We expect 200 class folders. If we have 0 subdirs, something is wrong.
        has_subdirs = any(x.is_dir() for x in val_dir.iterdir())
        if not has_subdirs:
            raise RuntimeError(
                f"Validation set at {val_dir} seems corrupted (no class folders found)."
            )
