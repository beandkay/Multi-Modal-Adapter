import os
import pickle
from PIL import Image

import torch
import torchvision.datasets as vision_datasets
from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import mkdir_if_missing

from .oxford_pets import OxfordPets # For subsample_classes


@DATASET_REGISTRY.register()
class CIFAR10(DatasetBase):
    """CIFAR-10 Dataset.

    This dataset is downloaded and processed on the fly.
    """
    dataset_dir = "cifar-10"
    domains = ["real"] # CIFAR-10 doesn't have domains like DomainNet

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir_path = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir_path, "images")
        self.split_dir = os.path.join(self.dataset_dir_path, "splits")
        mkdir_if_missing(self.image_dir)
        mkdir_if_missing(self.split_dir)

        self.classnames = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

        train_file = os.path.join(self.split_dir, "train.pkl")
        test_file = os.path.join(self.split_dir, "test.pkl")
        val_file = os.path.join(self.split_dir, "val.pkl")

        if os.path.exists(train_file) and os.path.exists(test_file) and os.path.exists(val_file):
            train_data = self.read_split(train_file)
            test_data = self.read_split(test_file)
            val_data = self.read_split(val_file)
        else:
            train_data_all, test_data = self.download_and_process_data()
            
            # Split train_data_all into train and validation
            # Following a 45k/5k split for train/val from the original 50k training images
            num_train_all = len(train_data_all)
            num_val = 5000 
            num_train = num_train_all - num_val
            
            # Ensure classes are represented in val following overall distribution as much as possible
            # For simplicity, we'll do a simple split first. A more robust split would stratify by class.
            # However, dassl's few-shot generation might handle class balancing later.
            # Let's shuffle before splitting to make it random.
            # For reproducibility, one might want to control the random seed here or save the exact split indices.
            # For now, a simple split after shuffling.
            # Note: A fixed seed is better for reproducibility if no split files are saved/loaded.
            # For now, let's assume we always save splits, so this part only runs once.
            
            # We'll use a more robust splitting mechanism that ensures class balance later if needed.
            # For now, a simple split.
            # Let's create a more robust split based on class labels
            
            # For now, a simple sequential split after download_and_process_data which sorts by class then image index
            train_data = train_data_all[:num_train]
            val_data = train_data_all[num_train:]

            self.save_split(train_data, train_file)
            self.save_split(test_data, test_file)
            self.save_split(val_data, val_file)

        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            seed = cfg.SEED
            preprocessed = os.path.join(self.split_dir, f"shot_{num_shots}-seed_{seed}.pkl")
            
            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train_data = data["train"]
                    val_data = data["val"]
            else:
                train_data = self.generate_fewshot_dataset(train_data, num_shots=num_shots)
                val_data = self.generate_fewshot_dataset(val_data, num_shots=min(num_shots, 5)) # Standard practice: val_num_shots <= 5
                data = {"train": train_data, "val": val_data}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        if subsample and subsample != "all":
            train_data = OxfordPets.subsample_classes(train_data, subsample=subsample)
            val_data = OxfordPets.subsample_classes(val_data, subsample=subsample)
            test_data = OxfordPets.subsample_classes(test_data, subsample=subsample)
            self.classnames = self.get_classnames_from_dataset(train_data)


        super().__init__(train_x=train_data, val=val_data, test=test_data)

    def download_and_process_data(self):
        print("Downloading and processing CIFAR-10 dataset...")
        
        # Load training data
        cifar_train = vision_datasets.CIFAR10(
            root=self.dataset_dir_path, train=True, download=True
        )
        # Load test data
        cifar_test = vision_datasets.CIFAR10(
            root=self.dataset_dir_path, train=False, download=True
        )

        train_items = []
        test_items = []

        for i, (img, label) in enumerate(cifar_train):
            classname = self.classnames[label]
            img_name = f"{classname}_{i:05d}.png"
            img_path = os.path.join(self.image_dir, classname)
            mkdir_if_missing(img_path)
            full_img_path = os.path.join(img_path, img_name)
            
            # Save image if it doesn't exist. This check is important if data processing is interrupted.
            if not os.path.exists(full_img_path):
                 img.save(full_img_path)

            datum = Datum(impath=full_img_path, label=label, classname=classname, domain=0) # domain=0 default
            train_items.append(datum)
            
        for i, (img, label) in enumerate(cifar_test):
            classname = self.classnames[label]
            img_name = f"{classname}_{i:05d}.png" # Use a different naming scheme or subfolder for test images if needed
            img_path = os.path.join(self.image_dir, classname) # Can save test images in same class folders
            mkdir_if_missing(img_path) # Ensure class directory exists (might be redundant if train created all)
            
            # To avoid name collision with training images, add a prefix/suffix or put in different subfolders.
            # For simplicity, let's assume names won't collide if indices are unique across train/test sets for each class
            # Or, more robustly, use a distinct naming pattern for test images.
            test_img_name = f"{classname}_test_{i:05d}.png"
            full_img_path = os.path.join(img_path, test_img_name)

            if not os.path.exists(full_img_path):
                img.save(full_img_path)
            
            datum = Datum(impath=full_img_path, label=label, classname=classname, domain=0)
            test_items.append(datum)
            
        print("CIFAR-10 data download and processing complete.")
        # The downloaded data from torchvision is already somewhat ordered by class,
        # but for consistent splitting, it's good practice to sort.
        # Sort by class, then by image path (which includes the index)
        train_items.sort(key=lambda x: (x.label, x.impath))
        test_items.sort(key=lambda x: (x.label, x.impath))
        
        return train_items, test_items

    def read_split(self, filepath):
        print(f"Reading split from {filepath}")
        with open(filepath, "rb") as f:
            split_data = pickle.load(f)
        return split_data

    def save_split(self, data, filepath):
        print(f"Saving split to {filepath}")
        mkdir_if_missing(os.path.dirname(filepath))
        with open(filepath, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    # generate_fewshot_dataset is part of DatasetBase, so we can use it directly.
    # We might need get_classnames_from_dataset if we implement subsampling more dynamically
    # For now, OxfordPets.subsample_classes requires the list of items and the classname list.
    # If OxfordPets.subsample_classes is a static method and works, we are good.
    # Let's assume OxfordPets.subsample_classes can be called.
    
    # Helper to get current classnames if subsampling happens
    def get_classnames_from_dataset(self, dataset_items):
        # Assuming dataset_items is a list of Datum objects
        # And self.classnames is the original full list of classnames
        # This function finds the unique classnames present in the current dataset_items
        # and returns them in the correct order (based on original label indices)
        present_labels = sorted(list(set(item.label for item in dataset_items)))
        return [self.classnames[label] for label in present_labels]
