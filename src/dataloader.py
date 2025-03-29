import torch
import glob
import os 
from PIL import Image
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from src.processor import Samprocessor
from src.segment_anything import build_sam_vit_b, SamPredictor
from src.lora import LoRA_sam
import src.utils as utils
import yaml


class DatasetSegmentation(Dataset):
    """
    Dataset to process the images and masks

    Arguments:
        folder_path (str): The path of the folder containing the images
        processor (obj): Samprocessor class that helps pre processing the image, and prompt 
    
    Return:
        (dict): Dictionnary with 4 keys (image, original_size, boxes, ground_truth_mask)
            image: image pre processed to 1024x1024 size
            original_size: Original size of the image before pre processing
            boxes: bouding box after adapting the coordinates of the pre processed image
            ground_truth_mask: Ground truth mask
    """

    def __init__(self, config_file: dict, processor: Samprocessor, mode: str):
        super().__init__()
        if mode == "train":
            self.img_files = glob.glob(os.path.join(config_file["DATASET"]["TRAIN_PATH"],'images','*.jpg'))
            self.mask_files = []
            for img_path in self.img_files:
                self.mask_files.append(os.path.join(config_file["DATASET"]["TRAIN_PATH"],'masks', os.path.basename(img_path)[:-4] + ".jpg")) 

        else:
            self.img_files = glob.glob(os.path.join(config_file["DATASET"]["TEST_PATH"],'images','*.jpg'))
            self.mask_files = []
            for img_path in self.img_files:
                self.mask_files.append(os.path.join(config_file["DATASET"]["TEST_PATH"],'masks', os.path.basename(img_path)[:-4] + ".jpg"))


        self.processor = processor

    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, index: int) -> list:
            img_path = self.img_files[index]
            mask_path = self.mask_files[index]
            # get image and mask in PIL format
            image =  Image.open(img_path)
            mask = Image.open(mask_path)
            mask = mask.convert('1')
            ground_truth_mask =  np.array(mask)
            original_size = tuple(image.size)[::-1]
    
            # get bounding box prompt
            box = utils.get_bounding_box(ground_truth_mask)
            inputs = self.processor(image, original_size, box)
            inputs["ground_truth_mask"] = torch.from_numpy(ground_truth_mask)

            return inputs
    
import torch.nn.functional as F

def collate_fn(batch):
    """
    Collate function that resizes all ground truth masks to a fixed size.
    Ensures that all tensors in the batch have the same shape.
    
    Arguments:
        batch (list of dict): List of samples from DatasetSegmentation.
    
    Returns:
        list(dict): The transformed batch.
    """
    target_size = (1024, 1024)  # Ajusta este tamaño según lo que necesites

    for sample in batch:
        # Agregar dimensión del canal (C=1)
        mask = sample["ground_truth_mask"].unsqueeze(0).unsqueeze(0).float()  # (1, H, W) → (1, 1, H, W)
        
        # Redimensionar la máscara
        mask_resized = F.interpolate(mask, size=target_size, mode='nearest')
        
        # Quitar la dimensión extra de batch y canal
        sample["ground_truth_mask"] = mask_resized.squeeze(0).squeeze(0)  # (1, 1, H, W) → (H, W)

    return batch  # Devuelve la lista con máscaras normalizadas
