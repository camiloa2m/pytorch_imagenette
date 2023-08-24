""" Module to download Imagenette and generate pytorch dataset """

import os
import requests
import tarfile
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image


def download_imagenette(url, local_path="./"):
    """
    Args:
        url (_type_): url where the imagenette file is hosted.
        local_path (str, optional): Path where you want to host
                                    the images dataset. Defaults to "./".
    """
    # Select the name of the file to download
    url_path_file = url.split('/')[-1]

    # Default original folder name
    foldername = os.path.splitext(url_path_file)[0]
    final_path = os.path.join(local_path, foldername)
    if not os.path.exists(final_path):
        # Download the file
        if not os.path.exists(url_path_file):
            print("Downloading", url, "...")
            response = requests.get(url)
            open(url_path_file, "wb").write(response.content)
        # Extract files
        print("Starting", url_path_file, "files extraction...")
        file = tarfile.open(url_path_file)
        file.extractall(path=local_path)
        file.close()
        print("Files extraction completed in:", final_path)
        # Remove downloaded file
        os.remove(url_path_file)

    else:
        print(foldername, "folder already exists.")
        print("You can find it in:", final_path)


class Imagenette(Dataset):
    def __init__(self,
                 annotations_file,
                 img_dir,
                 labels_col=1,
                 train=True,
                 shuffle=True,
                 transform=None,
                 target_transform=True):
        """Imagenette Dataset.

        Args:
            annotations_file (string): Root directory of the CSV file that
                                stores the labels and path information.
            img_dir (string): Root directory of the imagenette dataset.
            labels_col (int, optional): Index of the column in the
                                annotations_file that represents the labels.
                                Defaults to 1.
            train (bool, optional): If True, creates dataset from training set;
                                otherwise creates from test set.
                                Defaults to True.
            shuffle (bool, optional): If True, the data is shuffled;
                                otherwise not.
                                Defaults to True.
            transform (callable, optional): A function/transformation that
                                takes a tensor image and returns a transformed
                                version. For example, transforms.RandomCrop.
                                Defaults to None.
            target_transform (bool, optional): Boolean that defines if the
                                original labels are transformed to integers,
                                or if they are left unmodified.
                                Defaults to True.
        """

        self.info_csv = pd.read_csv(annotations_file)
        self.labels_col = labels_col
        self.img_dir = img_dir

        if train:
            self.info_csv = self.info_csv[~ self.info_csv['is_valid']]
            self.info_csv = self.info_csv.reset_index(drop=True)
        else:
            self.info_csv = self.info_csv[self.info_csv['is_valid']]
            self.info_csv = self.info_csv.reset_index(drop=True)

        if shuffle:
            self.info_csv = self.info_csv.sample(frac=1)
            self.info_csv = self.info_csv.reset_index(drop=True)

        self.transform = transform

        # Transformation to convert original labels to integers
        self.old_lbs = self.info_csv.iloc[:, labels_col].unique()
        self.cat_to_num = dict(zip(self.old_lbs, range(len(self.old_lbs))))
        self.target_transform = None
        if target_transform:
            self.target_transform = lambda x: self.cat_to_num[x]

    def __len__(self):
        return len(self.info_csv)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.info_csv.iloc[idx, 0])
        image = Image.open(img_path)
        label = self.info_csv.iloc[idx, self.labels_col]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
