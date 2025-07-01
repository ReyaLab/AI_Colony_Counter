#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 14:23:27 2025

@author: mattc
"""

import os
import cv2

# file_list = os.listdir(path)
# file_list = [x for x in file_list if (x[-4::]==".tif" or x[-5::]==".tiff")]
def cut_img(path, x):
    img_map = {}
    img = cv2.imread(path + x)
    name = x.split(".")[0]
    i_num = img.shape[0]/512
    j_num = img.shape[1]/512
    count = 1
    for i in range(int(i_num)):
        for j in range(int(j_num)):
            img2 = img[(512*i):(512*(i+1)), (512*j):(512*(j+1))]
            cv2.imwrite(path+name+'_part'+str(count)+'.tif', img2)
            img_map[count] = path+name+'_part'+str(count)+'.tif'
            count +=1
    return(img_map)
    
import numpy as np

def stitch(img_map):
    for x in img_map:
        temp = img_map[x]
        img_map[x] = cv2.imread(temp)
        if (img_map[x] is None):
            img_map[x] = cv2.imread(temp, cv2.IMREAD_UNCHANGED)
        os.remove(temp)
    rows = [
        np.hstack([img_map[1], img_map[2], img_map[3], img_map[4]]),  # First row (images 0 to 3)
        np.hstack([img_map[5], img_map[6], img_map[7], img_map[8]]),  # Second row (images 4 to 7)
        np.hstack([img_map[9], img_map[10], img_map[11], img_map[12]])  # Third row (images 8 to 11)
    ]
    
    # Stack rows vertically
    return(np.vstack(rows))

#img_map = cut_img(path, file_list[0])


from PIL import Image



import matplotlib.pyplot as plt

def visualize_segmentation(mask, image=0):
    plt.figure(figsize=(10, 5))

    if(not np.isscalar(image)):
        # Show original image if it is entered
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



# Load image processor
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
image_processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b3-finetuned-cityscapes-1024-1024")

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")  # Open and convert to RGB
    inputs = image_processor(image, return_tensors="pt")  # Preprocess for model
    return image, inputs["pixel_values"]

def postprocess_mask(logits):
    mask = torch.argmax(logits, dim=1)  # Take argmax across the class dimension
    return mask.squeeze().cpu().numpy()  # Convert to NumPy array


def eval_img(image_path, model):
    # Load and preprocess image
    image, pixel_values = preprocess_image(image_path)
    pixel_values = pixel_values.to(device)
    with torch.no_grad():  # No gradient calculation for inference
        outputs = model(pixel_values=pixel_values)  # Run model
        logits = outputs.logits
    # Convert logits to segmentation mask
    segmentation_mask = postprocess_mask(logits)
    #visualize_segmentation(segmentation_mask,image)
    segmentation_mask = cv2.resize(segmentation_mask, (512, 512), interpolation=cv2.INTER_LINEAR_EXACT)
    return(segmentation_mask)


# for x in img_map:
#     mask = eval_img(img_map[x])
#     cv2.imwrite(img_map[x], mask)
# del mask,x
# p = stitch(img_map)
# visualize_segmentation(p)

# num_colony = np.count_nonzero(p == 1)  # Counts number of 1s
# num_necrosis = np.count_nonzero(p == 2)

# num_necrosis/num_colony

def find_colonies(mask, size_cutoff, circ_cutoff):
    binary_mask = np.where(mask == 1, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contoursf = []
    for x in contours:
        area = cv2.contourArea(x)
        if (area < size_cutoff):
            continue
        perimeter = cv2.arcLength(x, True)

        # Avoid division by zero
        if perimeter == 0:
            continue
        
        # Calculate circularity
        circularity = (4 * np.pi * area) / (perimeter ** 2)
        if circularity >= circ_cutoff:
            contoursf.append(x)
    return(contoursf)

def find_necrosis(mask):
    binary_mask = np.where(mask == 2, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return(contours)

# contour_image = np.zeros_like(p)
# contours =  find_necrosis(p)
# cv2.drawContours(contour_image, contours, -1, (255), 2)
# visualize_segmentation(contour_image)
import pandas as pd
def compute_centroid(contour):
    M = cv2.moments(contour)
    if M["m00"] == 0:  # Avoid division by zero
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy)


def contours_overlap_using_mask(contour1, contour2, image_shape=(1536, 2048)):
    """Check if two contours overlap using a bitwise AND mask."""
    import numpy as np
    import cv2
    mask1 = np.zeros(image_shape, dtype=np.uint8)
    mask2 = np.zeros(image_shape, dtype=np.uint8)


    # Draw each contour as a white shape on its respective mask
    cv2.drawContours(mask1, [contour1], -1, 255, thickness=cv2.FILLED)
    cv2.drawContours(mask2, [contour2], -1, 255, thickness=cv2.FILLED)


    # Compute bitwise AND to find overlapping regions
    overlap = cv2.bitwise_and(mask1, mask2)
    
    return np.any(overlap)

def analyze_colonies(mask, size_cutoff, circ_cutoff, img):
    colonies = find_colonies(mask, size_cutoff, circ_cutoff)
    necrosis = find_necrosis(mask)
    
    data = []
    
    for colony in colonies:
        colony_area = cv2.contourArea(colony)
        centroid = compute_centroid(colony)
        if colony_area <= 50:
            continue
        mask = np.zeros(img.shape, np.uint8)
        cv2.drawContours(mask, [colony], -1, 255, cv2.FILLED)
        pix = img[mask == 255]
        # Check if any necrosis contour is inside the colony
        necrosis_area = 0
        nec_list =[]
        for nec in necrosis:
            # Check if the first point of the necrosis contour is inside the colony
            if contours_overlap_using_mask(colony, nec):
                nec_area = cv2.contourArea(nec)
                necrosis_area += nec_area
                nec_list.append(nec)

        data.append({
            "colony_area": colony_area,
            "necrotic_area": necrosis_area,
            "centroid": centroid,
            "percent_necrotic": necrosis_area/colony_area,
            "contour": colony,
            "nec_contours": nec_list,
            'mean_pixel_value':np.mean(pix)
        })

    # Convert results to a DataFrame
    df = pd.DataFrame(data)
    df.index = range(1,len(df.index)+1)
    return(df)

import torch    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
def main(args):
    path = args[0]
    file = args[1]
    min_size = args[2]
    min_circ = args[3]
    colonies = {}
    img_map = cut_img(path, file)
    from transformers import SegformerForSemanticSegmentation
    # Load fine-tuned model
    model = SegformerForSemanticSegmentation.from_pretrained(args[4]+"Segformer_Colony_Counter_GP")  # Adjust pat
    model.to(device)
    model.eval()  # Set to evaluation mode
    for z in img_map:
        mask = eval_img(img_map[z], model)
        cv2.imwrite(img_map[z], mask)
    del mask,z
    p = stitch(img_map)
    colonies = analyze_colonies(p, min_size, min_circ, cv2.imread(path + file))
    if len(colonies) <=0:
    	caption = np.ones((150, 2048, 3), dtype=np.uint8) * 255  # Multiply by 255 to make it white
    	cv2.putText(caption, 'No colonies detected.', (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
    	img = cv2.imread(path + file)
    	file = file.split('.')[0]
    	cv2.imwrite(path+file+'.png', np.vstack((img, caption)))
    	return(np.vstack((img, caption)))
    
    img = cv2.imread(path + file)
    img = cv2.copyMakeBorder(img,top=0, bottom=10,left=0,right=10, borderType=cv2.BORDER_CONSTANT,  value=[255, 255, 255]) 
    
    
    colonies = colonies.sort_values(by=["colony_area"], ascending=False)
    colonies = colonies[colonies["colony_area"]>= min_size]
    colonies.index = range(1,len(colonies.index)+1) 
    
    for i in range(len(colonies)): 
        cv2.drawContours(img, [list(colonies["contour"])[i]], -1, (0, 255, 0), 2)
        cv2.drawContours(img, list(colonies['nec_contours'])[i], -1, (0, 0, 255), 2)
        coords = list(list(colonies["centroid"])[i])
        if coords[0] > 1950:
            #if a colony is too close to the right edge, makes the label move to left
            coords[0] = 1950
        cv2.putText(img, str(colonies.index[i]), tuple(coords), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2)
    img = cv2.copyMakeBorder(img,top=10, bottom=0,left=10,right=0, borderType=cv2.BORDER_CONSTANT,  value=[255, 255, 255]) 
    colonies = colonies.drop('contour', axis=1)
    colonies = colonies.drop('nec_contours', axis=1)
    
    colonies.insert(loc=0, column="Colony Number", value=[str(x) for x in range(1, len(colonies)+1)])
    total_area_dark = sum(colonies['necrotic_area'])
    total_area_light = sum(colonies['colony_area'])
    ratio = total_area_dark/(abs(total_area_light)+1)
    radii = [np.sqrt(x/3.1415) for x in list(colonies['colony_area'])]
    volumes = [4.189*(x**3) for x in radii]
    colonies['Colony volume'] = volumes
    del radii, volumes
    meanpix = sum(colonies['mean_pixel_value'] * colonies['colony_area'])/total_area_light
    colonies.loc[len(colonies)+1] = ["Total", total_area_light, total_area_dark, None, ratio,meanpix, sum(colonies['Colony volume'])]
    del meanpix
    colonies = colonies[["Colony Number", 'Colony volume', "colony_area",'mean_pixel_value', "centroid", "necrotic_area","percent_necrotic"]]
    Parameters = pd.DataFrame({"Minimum colony size in pixels":[min_size], "Minimum colony circularity":[min_circ]})
    file = file.split('.')[0]
    with pd.ExcelWriter(path+file+'.xlsx') as writer:
        colonies.to_excel(writer, sheet_name="Colony data", index=False)
        Parameters.to_excel(writer, sheet_name="Parameters", index=False)
    caption = np.ones((150, 2068, 3), dtype=np.uint8) * 255  # Multiply by 255 to make it white
    cv2.putText(caption, "Total area necrotic: "+str(total_area_dark)+ ", Total area living: "+str(total_area_light)+", Ratio: "+str(ratio), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
    cv2.putText(caption, "Total number of colonies: "+str(len(colonies)-1), (40, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)


    cv2.imwrite(path+file+'.png', np.vstack((img, caption)))

    return(np.vstack((img, caption)))

