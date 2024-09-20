import os
import sys
from cellpose import models, train, io
import tifffile
import torch
from torchmetrics import JaccardIndex, Dice, Precision, Recall
from torchmetrics.segmentation import GeneralizedDiceScore
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import albumentations as A
from sklearn.model_selection import train_test_split

def load_cellpose_modelpath(model_path: str, gpu: bool = True) -> models.CellposeModel:
    """Load a Cellpose model from a specified path."""
    print('Loading Cellpose Models from folder ...')
    return models.CellposeModel(gpu=gpu, pretrained_model=str(model_path))

def calculate_metrics(true_mask, masks_pred):
    """Calculate evaluation metrics for model predictions."""
    intersection = np.logical_and(true_mask, masks_pred)
    dice_score = 2. * intersection.sum() / (true_mask.sum() + masks_pred.sum())
    union = np.logical_or(true_mask, masks_pred)
    iou_score = intersection.sum() / union.sum() if union.sum() > 0 else 0
    pixel_accuracy = np.sum(true_mask == masks_pred) / true_mask.size
    num_rois_true = np.unique(true_mask).size - 1  # Exclude background
    num_rois_pred = np.unique(masks_pred).size - 1  # Exclude background
    return {
        "Dice Score": dice_score,
        "IoU Score": iou_score,
        "Pixel Accuracy": pixel_accuracy,
        "Number of True ROIs": num_rois_true,
        "Number of Predicted ROIs": num_rois_pred
    }

def main(model_name: str, retrain: bool):
    save_path = '/root/capsule/scratch/'
    folder = 'retrained' if retrain else 'default'
    model_save_path = os.path.join(save_path, model_name, folder)

    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    
    io.logger_setup()
    
    # Define paths
    data_dir = '/root/capsule/data/iGluSnFR_Soma_Annotation'
    
    # Collect all image and mask file paths
    image_files = sorted([f for f in os.listdir(data_dir) if f.endswith('_merged.tif')])
    mask_files = sorted([f for f in os.listdir(data_dir) if f.endswith('_segmented_v2.tif')])
    
    # Ensure that each image has a corresponding mask
    assert len(image_files) == len(mask_files), "Number of images and masks must match."
    
    # Load all images and masks
    images = [tifffile.imread(os.path.join(data_dir, img))[:, 1, :, :] for img in image_files]
    masks = [tifffile.imread(os.path.join(data_dir, msk)) for msk in mask_files]
    
    # Ensure images and masks have the same number of frames
    for img, msk in zip(images, masks):
        assert img.shape[0] == msk.shape[0], "Number of frames in images and masks must match."
    
    # Convert lists to numpy arrays
    images = np.concatenate(images, axis=0)
    masks_uint8 = np.concatenate(masks, axis=0).astype(np.uint8)
    
    # Normalize images to 0-1 range
    images = images.astype(np.float32) / 255.0
    
    # Define ranges for thresholds
    cellprob_thresholds = np.arange(-6, 7, 1)  # Range from -6 to 6
    flow_thresholds = np.arange(0.1, 3.1, 0.1)  # Range from 0.1 to 3.0

    if retrain:
        print('-------------Retraining cellpose with custom data-------------')
        # Split data into train+val and test sets initially (no augmentation yet)
        train_val_images, test_images, train_val_masks, test_masks = train_test_split(
            images, masks_uint8, test_size=0.15, random_state=42)
        
        # Split train+val into train and validation sets
        train_images, val_images, train_masks, val_masks = train_test_split(
            train_val_images, train_val_masks, test_size=0.176, random_state=42)  # 0.176 to make validation 15% of total data
        
        # Define an augmentation pipeline for training data only
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=45, p=0.5),
        ], is_check_shapes=False)
        
        augmented_images = []
        augmented_masks = []
        
        # Augment each training image multiple times
        num_augmentations = 5 # Number of times to augment each image
        for img, msk in zip(train_images, train_masks):
            for _ in range(num_augmentations):
                transformed = transform(image=img, mask=msk)
                augmented_images.append(transformed['image'])
                augmented_masks.append(transformed['mask'])
        
        # Convert lists to numpy arrays and combine with original training data
        augmented_images = np.array(augmented_images)
        augmented_masks = np.array(augmented_masks)
        train_images_combined = np.concatenate((train_images, augmented_images), axis=0)
        train_masks_combined = np.concatenate((train_masks, augmented_masks), axis=0)
        
        # Initialize Cellpose model
        model = models.CellposeModel(gpu=True, model_type=model_name)
        
        # Train the model (example; adjust parameters as needed)
        model_path, train_losses, test_losses = train.train_seg(
            model.net,
            train_data=train_images_combined,
            train_labels=train_masks_combined,
            test_data=val_images,
            test_labels=val_masks,
            channels=[0, 0], # Adjust channels if needed
            normalize=True,
            weight_decay=1e-4,
            SGD=False,
            learning_rate=0.1,
            n_epochs=2,
            save_path=model_save_path,
            model_name=f'{model_name}_cellpose_model.pth'
        )
        
        # Define epochs based on the number of training epochs
        epochs = range(1, len(train_losses) + 1)
        
        # Plotting the losses
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, train_losses, label='Training Loss', marker='o')
        plt.plot(epochs, test_losses, label='Testing Loss', marker='o')
        plt.title('Training and Testing Losses over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.xticks(epochs)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(model_save_path, f'{model_name}_loss.png'), bbox_inches='tight')
        plt.close()
        
        # Load saved model for evaluation on test set without augmentation applied during training or validation phases.
        print('model_path---------->', model_path)
        model = load_cellpose_modelpath(model_path)
        
        results_list = []
        # for i, (image, true_mask) in enumerate(zip(test_images, test_masks)):
        #     results = model.eval(image, channels=[0, 0])
        #     if len(results) == 3:
        #         masks_pred, flows, styles = results
        #     else:
        #         masks_pred, flows, styles, diams = results
            
        #     metrics = calculate_metrics(true_mask, masks_pred)
        #     results_list.append({
        #         'Image_Index': i,
        #         'Dice_Score': metrics['Dice Score'],
        #         'IoU_Score': metrics['IoU Score'],
        #         'Pixel_Accuracy': metrics['Pixel Accuracy'],
        #         'Number_of_True_ROIs': metrics['Number of True ROIs'],
        #         'Number_of_Predicted_ROIs': metrics['Number of Predicted ROIs']
        #     })
        
        # results_df = pd.DataFrame(results_list)
        # results_df['Model_Name'] = model_name
        # results_df.to_csv(os.path.join(save_path, f'{model_name}_results.csv'), index=False)
        
        random_indices = random.sample(range(len(test_images)), 20)
        for idx in random_indices:
            results = model.eval(test_images[idx], channels=[0, 0])
            if len(results) == 3:
                masks_pred, flows, styles = results
            else:
                masks_pred, flows, styles, diams = results
            
            metrics = calculate_metrics(test_masks[idx], masks_pred)

            results_list.append({
                'Image_Index': i,
                'Dice_Score': metrics['Dice Score'],
                'IoU_Score': metrics['IoU Score'],
                'Pixel_Accuracy': metrics['Pixel Accuracy'],
                'Number_of_True_ROIs': metrics['Number of True ROIs'],
                'Number_of_Predicted_ROIs': metrics['Number of Predicted ROIs']
            })

            results_df=pd.DataFrame(results_list)
            results_df['Model_Name'] = model_name
            
            results_df.to_csv(os.path.join(model_save_path,f'{model_name}_results.csv'),index=False)
            
            print(f"Evaluation completed. Results saved to {os.path.join(model_save_path,f'{model_name}_results.csv')}")
            
            fig, ax = plt.subplots(1, 4, figsize=(16, 6))
            fig.suptitle(title, fontsize=12)
            ax[0].imshow(test_images[idx])
            ax[0].set_title('Original Image')
            ax[0].axis('off')
            ax[1].imshow(test_images[idx], cmap='gray')
            ax[1].imshow(test_masks[idx], cmap='jet', alpha=0.5)
            ax[1].set_title('Ground Truth Mask')
            ax[1].axis('off')
            ax[2].imshow(test_masks[idx], cmap='gray')
            ax[2].imshow(masks_pred, cmap='jet', alpha=0.5)
            ax[2].set_title('Predicted Mask')
            ax[2].axis('off')
            ax[3].imshow(flows[0], cmap='gray')
            ax[3].set_title('Flow Field')
            ax[3].axis('off')
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(os.path.join(model_save_path, f'{model_name}_image_{idx}.png'), bbox_inches='tight')
            plt.close(fig)
    
    else:
        print('-------Using cellpose out of the box-------')
        # Logic for cellpose out of the box
        
        model = models.CellposeModel(gpu=True, model_type=model_name)
        
        results_list = []
        
        for i, (image, true_mask) in enumerate(zip(images, masks_uint8)):
            results = model.eval(image, channels=[0, 0])
            if len(results) == 3:
                masks_pred, flows, styles = results
            else:
                masks_pred, flows, styles, diams = results
            
            metrics = calculate_metrics(true_mask, masks_pred)
            results_list.append({
                'Image_Index': i,
                'Dice_Score': metrics['Dice Score'],
                'IoU_Score': metrics['IoU Score'],
                'Pixel_Accuracy': metrics['Pixel Accuracy'],
                'Number_of_True_ROIs': metrics['Number of True ROIs'],
                'Number_of_Predicted_ROIs': metrics['Number of Predicted ROIs']
            })
        
        results_df = pd.DataFrame(results_list)
        results_df['Model_Name'] = model_name
        results_df.to_csv(os.path.join(model_save_path, f'{model_name}_results.csv'), index=False)
        
        random_indices = random.sample(range(len(images)), 20)
        for idx in random_indices:
            results = model.eval(images[idx], channels=[0, 0])
            if len(results) == 3:
                masks_pred, flows, styles = results
            else:
                masks_pred, flows, styles, diams = results
            
            metrics = calculate_metrics(masks_uint8[idx], masks_pred)
            title = (f"Image_Index: {idx}, "
                    f"Dice_Score: {metrics['Dice Score']:.2f}, "
                    f"IoU_Score: {metrics['IoU Score']:.2f}, "
                    f"Pixel_Accuracy: {metrics['Pixel Accuracy']:.2f}, "
                    f"Number_of_True_ROIs: {metrics['Number of True ROIs']}, "
                    f"Number_of_Predicted_ROIs: {metrics['Number of Predicted ROIs']}")
            
            fig, ax = plt.subplots(1, 4, figsize=(16, 6))
            fig.suptitle(title, fontsize=12)
            ax[0].imshow(images[idx])
            ax[0].set_title('Original Image')
            ax[0].axis('off')
            ax[1].imshow(images[idx], cmap='gray')
            ax[1].imshow(masks_uint8[idx], cmap='jet', alpha=0.5)
            ax[1].set_title('Ground Truth Mask')
            ax[1].axis('off')
            ax[2].imshow(images[idx], cmap='gray')
            ax[2].imshow(masks_pred, cmap='jet', alpha=0.5)
            ax[2].set_title('Predicted Mask')
            ax[2].axis('off')
            ax[3].imshow(flows[0], cmap='gray')
            ax[3].set_title('Flow Field')
            ax[3].axis('off')
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(os.path.join(model_save_path, f'{model_name}_image_{idx}.png'), bbox_inches='tight')
            plt.close(fig)        
        print(f"Evaluation completed. Results saved to {os.path.join(model_save_path,f'{model_name}_results.csv')}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <model_name> <retrain>")
        sys.exit(1)

    model_name_arg = sys.argv[1]
    retrain_flag_arg = sys.argv[2].lower() == 'true'
    
    main(model_name_arg, retrain_flag_arg)