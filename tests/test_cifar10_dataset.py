import unittest
import os
import shutil
from argparse import Namespace

# Adjust the import path based on where the main project files are relative to the tests directory.
# If tests/ is at the root alongside 'datasets/', this might require sys.path manipulation
# or making the project installable. For now, assume it can find 'datasets'.
# A common pattern is to add the project root to sys.path in the test file.
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datasets.cifar10 import CIFAR10
from dassl.utils import mkdir_if_missing

class TestCIFAR10Dataset(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.temp_data_root = "./temp_cifar10_test_data"
        # Ensure the directory is clean before running tests
        if os.path.exists(cls.temp_data_root):
            shutil.rmtree(cls.temp_data_root)
        mkdir_if_missing(cls.temp_data_root)

        # Mock CFG object
        cls.cfg = Namespace()
        cls.cfg.DATASET = Namespace()
        cls.cfg.DATASET.ROOT = cls.temp_data_root
        cls.cfg.DATASET.NAME = "CIFAR10" # Important for the dataset class to find its dir
        cls.cfg.DATASET.NUM_SHOTS = -1  # Load full dataset
        cls.cfg.DATASET.SUBSAMPLE_CLASSES = "all"
        cls.cfg.SEED = 1 # Required by few-shot logic, even if not used for full dataset

        # Instantiate the dataset once for the class
        # This will trigger download and processing if not already done
        # and save images to disk within temp_data_root/cifar-10
        # The actual dataset path will be temp_data_root/cifar-10/
        cls.dataset = CIFAR10(cls.cfg)

    @classmethod
    def tearDownClass(cls):
        # Clean up: remove the temporary data root directory
        if os.path.exists(cls.temp_data_root):
            shutil.rmtree(cls.temp_data_root)

    def test_dataset_loaded(self):
        self.assertTrue(self.dataset.train_x, "Training data should be loaded.")
        self.assertTrue(self.dataset.val, "Validation data should be loaded.")
        self.assertTrue(self.dataset.test, "Test data should be loaded.")

    def test_dataset_properties(self):
        self.assertEqual(len(self.dataset.classnames), 10, "Should have 10 class names.")
        self.assertEqual(self.dataset.num_classes, 10, "Should have 10 classes.")
        self.assertIn("airplane", self.dataset.classnames, "Class 'airplane' should be present.")
        self.assertIn("truck", self.dataset.classnames, "Class 'truck' should be present.")

    def test_data_splits_counts(self):
        # CIFAR-10 has 50000 training images, 10000 test images.
        # Our script splits training into 45000 for train_x, 5000 for val.
        self.assertEqual(len(self.dataset.train_x), 45000, "Default train split should have 45000 images.")
        self.assertEqual(len(self.dataset.val), 5000, "Default val split should have 5000 images.")
        self.assertEqual(len(self.dataset.test), 10000, "Test split should have 10000 images.")

if __name__ == '__main__':
    unittest.main()
