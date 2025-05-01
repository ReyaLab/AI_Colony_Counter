#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 16:20:16 2025

@author: mattc

This is to further train the ternary segformer (classifies background, colony, necrosis) with additional training data

"""

from PIL import Image
import numpy as np




import matplotlib.pyplot as plt

def visualize_segmentation(image, mask):
    plt.figure(figsize=(10, 5))

    # Show original image
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis("off")

    # Show segmentation mask
    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap="gray")  # Show as grayscale
    plt.title("Segmentation Mask")
    plt.axis("off")

    plt.show()


from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor
from torch.utils.data import DataLoader, Dataset
import torch
import os
class CellSegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, feature_extractor):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.feature_extractor = feature_extractor
        self.image_filenames = os.listdir(image_paths)
    
    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
       # Get image file path
       image_filename = self.image_filenames[idx]
       image_path = os.path.join(self.image_paths, image_filename)
       
       # Get corresponding mask file path by prepending 'Mask_'
       mask_filename = "Mask_" + image_filename
       if mask_filename[10].isdigit():
           mask_filename = mask_filename[:11] + 'tern' +mask_filename[11::]
       else:
           mask_filename = mask_filename[:10] + 'tern' +mask_filename[10::]
       mask_path = os.path.join(self.mask_paths, mask_filename)
       
       # Open the image and mask
       image = Image.open(image_path).convert("RGB")  # Convert to RGB (3 channels)
       mask = Image.open(mask_path).convert('L')      # Grayscale mask
       
       # Process the mask (convert 255 to 1 for foreground)
       mask = np.array(mask)
       
       # Extract features from the image (resize, normalize)
       encoded_inputs = self.feature_extractor(image, return_tensors="pt")
       
       # Return pixel values and the mask
       return {
           'pixel_values': encoded_inputs['pixel_values'].squeeze(),  # Remove extra dimension
           'labels': torch.tensor(mask, dtype=torch.long)
       }
image_paths = '/home/mattc/Documents/ColonyAssaySegformer/Mari_trainingset2/images_cropped/' 
mask_paths = '/home/mattc/Documents/ColonyAssaySegformer/Mari_trainingset2/labels_cropped/' 
# Example feature extractor and dataset setup
feature_extractor = SegformerFeatureExtractor(do_resize=True, size=512, resample=Image.BILINEAR)
train_dataset = CellSegmentationDataset(image_paths, mask_paths, feature_extractor)
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

model = SegformerForSemanticSegmentation.from_pretrained("/home/mattc/Documents/ColonyAssaySegformer/segformer_colony_model_ternary")  # Adjust path

# If using a device (e.g., GPU), make sure to move the model to the correct device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# Training Loop (basic setup with Hugging Face)
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="/home/mattc/Documents/ColonyAssaySegformer/results",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    learning_rate=5e-5,
    logging_dir="/home/mattc/Documents/ColonyAssaySegformer/logs",
    logging_steps=10,
    logging_strategy="steps",  # Log per step instead of per epoch
    report_to="none",  # Avoid logging to WandB/HuggingFace Hub
    disable_tqdm=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=train_dataset,  # You should split your data into training/validation sets
)
torch.cuda.empty_cache()
trainer.train()

trainer.save_model("/home/mattc/Documents/ColonyAssaySegformer/segformer_colony_model_ternary_finished")


# Load fine-tuned model
model = SegformerForSemanticSegmentation.from_pretrained("/home/mattc/Documents/ColonyAssaySegformer/segformer_colony_model_ternary")  # Adjust path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()  # Set to evaluation mode

# Load image processor
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
image_processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b3-finetuned-cityscapes-1024-1024")

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")  # Open and convert to RGB
    inputs = image_processor(image, return_tensors="pt")  # Preprocess for model
    return image, inputs["pixel_values"]

# Load and preprocess image
image_path = "/home/mattc/Documents/ColonyAssaySegformer/test_cropped/Z-Stack - TRANS - 05_part6.tif"  # Change this to your image path
image, pixel_values = preprocess_image(image_path)
pixel_values = pixel_values.to(device)

with torch.no_grad():  # No gradient calculation for inference
    outputs = model(pixel_values=pixel_values)  # Run model
    logits = outputs.logits
    
def postprocess_mask(logits):
    mask = torch.argmax(logits, dim=1)  # Take argmax across the class dimension
    return mask.squeeze().cpu().numpy()  # Convert to NumPy array

# Convert logits to segmentation mask
segmentation_mask = postprocess_mask(logits)

visualize_segmentation(image, segmentation_mask)