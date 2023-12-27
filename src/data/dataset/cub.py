import os
from pathlib import Path
import pandas as pd
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset


class Cub2011(Dataset):
    """
    Taken from: https://github.com/TDeVries/cub2011_dataset
    """

    base_folder = "CUB_200_2011/images"
    url = (
        "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1"
    )
    filename = "CUB_200_2011.tgz"
    tgz_md5 = "97eceeb196236b17998738112f37df78"

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform=None,
        loader=default_loader,
        download: bool = True,
        num_classes: int = 200,
    ):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.loader = default_loader
        self.train = train

        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted."
                + " You can use download=True to download it"
            )

        # class labels are from 1 to 200, shift to start at zero
        self.data["target"] = self.data["target"].apply(lambda x: x - 1)

        self._extract_mapping()
        self._fix_class_names()

    def _load_metadata(self):
        images = pd.read_csv(
            os.path.join(self.root, "CUB_200_2011", "images.txt"),
            sep=" ",
            names=["img_id", "filepath"],
        )
        image_class_labels = pd.read_csv(
            os.path.join(self.root, "CUB_200_2011", "image_class_labels.txt"),
            sep=" ",
            names=["img_id", "target"],
        )
        train_test_split = pd.read_csv(
            os.path.join(self.root, "CUB_200_2011", "train_test_split.txt"),
            sep=" ",
            names=["img_id", "is_training_img"],
        )

        data = images.merge(image_class_labels, on="img_id")
        self.data = data.merge(train_test_split, on="img_id")

        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def _download(self):
        import tarfile

        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = sample.target
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        return {
            "image": img,
            "label": target,
        }

    def _extract_mapping(self):
        folder_names = [str(Path(path).parent) for path in self.data.filepath]
        self.idx_to_class = {}
        for folder_name in folder_names:
            idx, class_name = folder_name.split(".")
            # CUB counts from 1, so shuft to 0
            self.idx_to_class[int(idx) - 1] = class_name.replace("_", " ")

    def _fix_class_names(self):
        # fix the class names
        # Note, proper lower and upper case are important!
        fix = {
            "Chuck will Widow": "Chuck will's Widow",
            "Scott Oriole": "Scott's Oriole",
            "Sayornis": "Sayornis Phoebe",
            "Baird Sparrow": "Baird's Sparrow",
            "Brewer Blackbird": "Brewer's Blackbird",
            "Brandt Cormorant": "Brandt's Cormorant",
            "Heermann Gull": "Heermann's Gull",
            "Clark Nutcracker": "Clark's Nutcracker",
            "Brewer Sparrow": "Brewer's Sparrow",
            "Harris Sparrow": "Harris's Sparrow",
            "Henslow Sparrow": "Henslow's Sparrow",
            "Le Conte Sparrow": "Le Conte's Sparrow",
            "Lincoln Sparrow": "Lincoln's Sparrow",
            "Nelson Sharp tailed Sparrow": "Nelson's Sharp tailed Sparrow",
            "Forsters Tern": "Forster's Tern",
            "Swainson Warbler": "Swainson's Warbler",
            "Wilson Warbler": "Wilson's Warbler",
            "Bewick Wren": "Bewick's Wren",
        }
        fixed_class_names = {
            idx: fix.get(class_name, class_name)
            for idx, class_name in self.idx_to_class.items()
        }
        self.idx_to_class = fixed_class_names


if __name__ == "__main__":
    _ = Cub2011(root="/mnt/character-based-classification/data/CUB")
    ffoo = 9
