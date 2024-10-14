import os
import sys
from cellpose import models, io
import tifffile
import numpy as np
import torch
import pandas as pd
import random
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from cellpose import metrics as cp_metrics

from stardist import matching

def calculate_metrics(true_mask, pred_mask, thresh=0.5):
    # Use the matching function from stardist to calculate metrics
    match = matching.matching(true_mask, pred_mask, thresh=thresh)
    
    return {
        "Dice Score": match.f1,
        "IoU Score": match.mean_matched_score,
        "Pixel Accuracy": match.accuracy,
        "Number of True ROIs": match.n_true,
        "Number of Predicted ROIs": match.n_pred,
        "Precision": match.precision,
        "Recall": match.recall,
        "F1 Score": match.f1,
        "True Positives": match.tp,
        "False Positives": match.fp,
        "False Negatives": match.fn,
        # 'True Negatives' is not directly provided by stardist.matching
        # It can be computed if needed by considering all pixels minus TP, FP, FN
        # "True Negatives": total_pixels - (match.tp + match.fp + match.fn),
        "Mean True Score": match.mean_true_score,
        "Mean Matched Score": match.mean_matched_score,
        "Panoptic Quality": match.panoptic_quality
    }

# def calculate_metrics(true_mask, pred_mask):
#     true_mask = true_mask.flatten() # For IOU flatten into  binary 
#     pred_mask = pred_mask.flatten()
    
#     precision = precision_score(true_mask, pred_mask, average='macro', zero_division=0)
#     recall = recall_score(true_mask, pred_mask, average='macro', zero_division=0)
#     f1 = f1_score(true_mask, pred_mask, average='macro', zero_division=0)
#     iou_score = jaccard_score(true_mask, pred_mask, average='macro', zero_division=0)
    
#     pixel_accuracy = np.mean(true_mask == pred_mask)
    
#     num_rois_true = len(np.unique(true_mask)) - 1  # Subtract 1 to exclude background
#     num_rois_pred = len(np.unique(pred_mask)) - 1  # Subtract 1 to exclude background
    
#     # Calculate true positives, false positives, false negatives, and true negatives for ROC curve
#     true_positives = np.sum((true_mask == 1) & (pred_mask == 1))
#     false_positives = np.sum((true_mask == 0) & (pred_mask == 1))
#     false_negatives = np.sum((true_mask == 1) & (pred_mask == 0))
#     true_negatives = np.sum((true_mask == 0) & (pred_mask == 0))
    
#     return {
#         "Dice Score": f1,
#         "IoU Score": iou_score,
#         "Pixel Accuracy": pixel_accuracy,
#         "Number of True ROIs": num_rois_true,
#         "Number of Predicted ROIs": num_rois_pred,
#         "Precision": precision,
#         "Recall": recall,
#         "F1 Score": f1,
#         "True Positives": true_positives,
#         "False Positives": false_positives,
#         "False Negatives": false_negatives,
#         "True Negatives": true_negatives
#     }

def process_image(i, image, true_mask, model, cellprob_thresholds, flow_thresholds):
    results_list = []
    print('image--------->', image.shape)
    for cellprob_threshold in cellprob_thresholds:
        for flow_threshold in flow_thresholds:
            results = model.eval(image,diameter=70, cellprob_threshold=cellprob_threshold, flow_threshold=flow_threshold)
            if len(results) == 3:
                masks_pred, flows, styles = results
            else:
                masks_pred, flows, styles, diams = results
                
            metrics = calculate_metrics(true_mask, masks_pred)
            results_list.append({
                'Image_Index': i,
                'Cellprob_Threshold': cellprob_threshold,
                'Flow_Threshold': flow_threshold,
                'Dice_Score': metrics['Dice Score'],
                'IoU_Score': metrics['IoU Score'],
                'Pixel_Accuracy': metrics['Pixel Accuracy'],
                'Number_of_True_ROIs': metrics['Number of True ROIs'],
                'Number_of_Predicted_ROIs': metrics['Number of Predicted ROIs'],
                'Precision': metrics['Precision'],
                'Recall': metrics['Recall'],
                'F1_Score': metrics['F1 Score'],
                'True_Positives': metrics['True Positives'],
                'False_Positives': metrics['False Positives'],
                'False_Negatives': metrics['False Negatives'],
                # 'True_Negatives' is not directly provided by stardist.matching
                # If needed, it can be computed separately
                # 'True_Negatives': metrics['True Negatives'],
                'Mean_True_Score': metrics['Mean True Score'],
                'Mean_Matched_Score': metrics['Mean Matched Score'],
                'Panoptic_Quality': metrics['Panoptic Quality']
            })

    return results_list

def main(model_name: str, retrain: bool, gpu_id):
    save_path = '/root/capsule/scratch/'
    folder = 'retrained' if retrain else 'default'
    model_save_path = os.path.join(save_path, model_name, folder)

    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    io.logger_setup()
    
    data_dir = '/root/capsule/data/iGluSnFR_Soma_Annotation'
    
    image_files = sorted([f for f in os.listdir(data_dir) if f.endswith('_merged.tif')])
    mask_files = sorted([f for f in os.listdir(data_dir) if f.endswith('_segmented_v2.tif')])
    
    assert len(image_files) == len(mask_files), "Number of images and masks must match."
    
    images = [tifffile.imread(os.path.join(data_dir, img))[:, 1, :, :] for img in image_files]
    masks = [tifffile.imread(os.path.join(data_dir, msk)).astype(np.uint8) for msk in mask_files]
    
    for img, msk in zip(images, masks):
        assert img.shape[0] == msk.shape[0], "Number of frames in images and masks must match."

    # Convert lists to numpy arrays
    images = np.concatenate(images, axis=0)
    masks_uint8 = np.concatenate(masks, axis=0).astype(np.uint8)
    
    # images = np.array(images).astype(np.float32) / 255.0
    
    cellprob_thresholds = np.arange(-6, 7, 2)
    flow_thresholds = np.arange(0.1, 3.1, 0.2)

    print('-------Using cellpose out of the box-------')
    
    model = models.CellposeModel(gpu=True, model_type=model_name, device=torch.device(f"cuda:{gpu_id}"))

    results_list = []
    
    total_images = len(images)
    
    lock = threading.Lock()
    
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_image, i, image, true_mask, model, cellprob_thresholds, flow_thresholds): i 
                   for i, (image, true_mask) in enumerate(zip(images, masks_uint8))}
        
        completed_count = 0
        
        for future in as_completed(futures):
            results_list.extend(future.result())
            with lock:
                completed_count += 1
                remaining_images = total_images - completed_count
                print(f'Remaining images: {remaining_images}')

    results_df = pd.DataFrame(results_list)
    results_df['Model_Name'] = model_name
    results_df.to_csv(os.path.join(model_save_path, f'{model_name}_threshold_results.csv'), index=False)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py <model_name> <retrain>")
        sys.exit(1)
    print('sys.argv',sys.argv)
    model_name_arg = sys.argv[1]
    retrain_flag_arg = sys.argv[2].lower() == 'true'
    gpu_id = int(sys.argv[3])
    
    main(model_name_arg, retrain_flag_arg, gpu_id)