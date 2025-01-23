import os
import sys
from cellpose import models, train, io
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

import matplotlib.pyplot as plt
import albumentations as A
from sklearn.model_selection import train_test_split


def load_cellpose_modelpath(model_path: str, gpu) -> models.CellposeModel:
    """Load a Cellpose model from a specified path."""
    print("Loading Cellpose Models from folder ...")
    return models.CellposeModel(device=gpu, pretrained_model=str(model_path))


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
        "Panoptic Quality": match.panoptic_quality,
    }


def process_image(
    i,
    image,
    true_mask,
    model,
    cellprob_thresholds,
    flow_thresholds,
    model_save_path,
    model_name,
):
    results_list = []
    for cellprob_threshold in cellprob_thresholds:
        for flow_threshold in flow_thresholds:
            results = model.eval(
                image,
                cellprob_threshold=cellprob_threshold,
                flow_threshold=flow_threshold,
            )
            if len(results) == 3:
                masks_pred, flows, styles = results
            else:
                masks_pred, flows, styles, diams = results

            metrics = calculate_metrics(true_mask, masks_pred)
            results_list.append(
                {
                    "Image_Index": i,
                    "Cellprob_Threshold": cellprob_threshold,
                    "Flow_Threshold": flow_threshold,
                    "Dice_Score": metrics["Dice Score"],
                    "IoU_Score": metrics["IoU Score"],
                    "Pixel_Accuracy": metrics["Pixel Accuracy"],
                    "Number_of_True_ROIs": metrics["Number of True ROIs"],
                    "Number_of_Predicted_ROIs": metrics["Number of Predicted ROIs"],
                    "Precision": metrics["Precision"],
                    "Recall": metrics["Recall"],
                    "F1_Score": metrics["F1 Score"],
                    "True_Positives": metrics["True Positives"],
                    "False_Positives": metrics["False Positives"],
                    "False_Negatives": metrics["False Negatives"],
                    "Mean_True_Score": metrics["Mean True Score"],
                    "Mean_Matched_Score": metrics["Mean Matched Score"],
                    "Panoptic_Quality": metrics["Panoptic Quality"],
                }
            )
            title = (
                f"Image_Index: {i}, "
                f"Cellprob_Threshold: {cellprob_threshold}, "
                f"Flow_Threshold: {flow_threshold}, "
                f"Panoptic_Quality: {metrics['Panoptic Quality']:.2f}, "
                f"IoU_Score: {metrics['IoU Score']:.2f}, "
                f"Pixel_Accuracy: {metrics['Pixel Accuracy']:.2f}, "
                f"Precision: {metrics['Precision']:.2f}, "
                f"Recall: {metrics['Recall']:.2f}, "
                f"Number_of_True_ROIs: {metrics['Number of True ROIs']}, "
                f"Number_of_Predicted_ROIs: {metrics['Number of Predicted ROIs']}"
            )
            fig, ax = plt.subplots(1, 4, figsize=(16, 6))
            fig.suptitle(title, fontsize=12)
            ax[0].imshow(image)
            ax[0].set_title("Original Image")
            ax[0].axis("off")
            ax[1].imshow(image, cmap="gray")
            ax[1].imshow(true_mask, cmap="jet", alpha=0.5)
            ax[1].set_title("Ground Truth Mask")
            ax[1].axis("off")
            ax[2].imshow(image, cmap="gray")
            ax[2].imshow(masks_pred, cmap="jet", alpha=0.5)
            ax[2].set_title("Predicted Mask")
            ax[2].axis("off")
            ax[3].imshow(flows[0], cmap="gray")
            ax[3].set_title("Flow Field")
            ax[3].axis("off")
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(
                os.path.join(
                    model_save_path,
                    f"{model_name}_image_{i}_cellprob_{cellprob_threshold}_flow_{flow_threshold}.png",
                ),
                bbox_inches="tight",
            )
            plt.close(fig)
    return results_list


def main(model_name: str, retrain: bool, gpu_id):
    save_path = "/root/capsule/scratch/"
    folder = "retrained" if retrain else "default"
    model_save_path = os.path.join(save_path, model_name, folder)

    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    io.logger_setup()

    data_dir = "/root/capsule/data/iGluSnFR_Soma_Annotation"

    image_files = sorted([f for f in os.listdir(data_dir) if f.endswith("_merged.tif")])
    mask_files = sorted(
        [f for f in os.listdir(data_dir) if f.endswith("_segmented_v2.tif")]
    )

    assert len(image_files) == len(mask_files), "Number of images and masks must match."

    images = [
        tifffile.imread(os.path.join(data_dir, img))[:, 1, :, :] for img in image_files
    ]
    masks = [
        tifffile.imread(os.path.join(data_dir, msk)).astype(np.uint8)
        for msk in mask_files
    ]

    for img, msk in zip(images, masks):
        assert (
            img.shape[0] == msk.shape[0]
        ), "Number of frames in images and masks must match."

    # Convert lists to numpy arrays
    images = np.concatenate(images, axis=0)
    masks_uint8 = np.concatenate(masks, axis=0).astype(np.uint8)

    # images = np.array(images).astype(np.float32) / 255.0
    if not retrain:
        cellprob_thresholds = np.arange(-6, 7, 2)
        flow_thresholds = np.arange(0.1, 3.1, 0.2)

        print("-------Using cellpose out of the box-------")

        model = models.CellposeModel(
            gpu=True, model_type=model_name, device=torch.device(f"cuda:{gpu_id}")
        )

        results_list = []

        total_images = len(images)

        lock = threading.Lock()

        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(
                    process_image,
                    i,
                    image,
                    true_mask,
                    model,
                    cellprob_thresholds,
                    flow_thresholds,
                    model_save_path,
                    model_name,
                ): i
                for i, (image, true_mask) in enumerate(zip(images, masks_uint8))
            }

            completed_count = 0

            for future in as_completed(futures):
                results_list.extend(future.result())
                with lock:
                    completed_count += 1
                    remaining_images = total_images - completed_count
                    print(f"Remaining images: {remaining_images}")

        results_df = pd.DataFrame(results_list)
        results_df["Model_Name"] = model_name
        results_df.to_csv(
            os.path.join(model_save_path, f"{model_name}_threshold_results.csv"),
            index=False,
        )

    else:
        print("-------Retraining Cellpose with SLAP2 data-------")
        # Split data into train+val and test sets initially (no augmentation yet)
        train_val_images, test_images, train_val_masks, test_masks = train_test_split(
            images, masks_uint8, test_size=0.15, random_state=42
        )

        # Split train+val into train and validation sets
        train_images, val_images, train_masks, val_masks = train_test_split(
            train_val_images, train_val_masks, test_size=0.176, random_state=42
        )  # 0.176 to make validation 15% of total data

        # Define an augmentation pipeline for training data only
        transform = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=45, p=0.5),
                A.GaussNoise(var_limit=(10.0, 50.0), mean=0, p=1.0),
            ],
            is_check_shapes=False,
        )

        augmented_images = []
        augmented_masks = []

        # Augment each training image multiple times
        num_augmentations = 5  # Number of times to augment each image
        for img, msk in zip(train_images, train_masks):
            for _ in range(num_augmentations):
                transformed = transform(image=img, mask=msk)
                augmented_images.append(transformed["image"])
                augmented_masks.append(transformed["mask"])

        # Convert lists to numpy arrays and combine with original training data
        augmented_images = np.array(augmented_images)
        augmented_masks = np.array(augmented_masks)
        train_images_combined = np.concatenate((train_images, augmented_images), axis=0)
        train_masks_combined = np.concatenate((train_masks, augmented_masks), axis=0)

        # Initialize Cellpose model
        model = models.CellposeModel(
            gpu=True, model_type=model_name, device=torch.device(f"cuda:{gpu_id}")
        )

        # Train the model (example; adjust parameters as needed)
        model_path, train_losses, test_losses = train.train_seg(
            model.net,
            train_data=train_images_combined,
            train_labels=train_masks_combined,
            test_data=val_images,
            test_labels=val_masks,
            channels=[0, 0],  # Adjust channels if needed
            normalize=True,
            weight_decay=1e-4,
            SGD=False,
            learning_rate=0.1,
            n_epochs=1000,
            save_path=model_save_path,
            model_name=f"{model_name}_cellpose_model.pth",
        )

        # Define epochs based on the number of training epochs
        epochs = range(1, len(train_losses) + 1)

        # Plotting the losses
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, train_losses, label="Training Loss", marker="o")
        plt.plot(epochs, test_losses, label="Validation Loss", marker="o")
        plt.title("Training and Validation Losses over Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.xticks(epochs)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(model_save_path, f"{model_name}_loss.png"), bbox_inches="tight"
        )
        plt.close()

        print("-------Testing Cellpose with SLAP2 data-------")

        # thresholds = {
        #     "neurips_grayscale_cyto2": (0.0, 0.3),
        #     "cyto": (0.0, 0.5),
        #     "nuclei": (0.0, 0.7),
        #     "cyto2": (2.0, 0.5),
        #     "cyto3": (2.0, 0.1),
        #     "cyto2_cp3": (2.0, 0.3)
        # } # Hardcode values based on out of the box tests.

        # cellprob_thresholds, flow_thresholds = thresholds.get(model_name, (None, None))

        cellprob_thresholds = np.arange(-1, 3, 1)
        flow_thresholds = np.arange(0.1, 3.1, 0.2)

        model = load_cellpose_modelpath(model_path, torch.device(f"cuda:{gpu_id}"))

        results_list = []

        total_images = len(test_images)

        lock = threading.Lock()

        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(
                    process_image,
                    i,
                    test_images,
                    test_masks,
                    model,
                    cellprob_thresholds,
                    flow_thresholds,
                    model_save_path,
                    model_name,
                ): i
                for i, (test_images, test_masks) in enumerate(
                    zip(test_images, test_masks)
                )
            }

            completed_count = 0

            for future in as_completed(futures):
                results_list.extend(future.result())
                with lock:
                    completed_count += 1
                    remaining_images = total_images - completed_count
                    print(f"Remaining images: {remaining_images}")

        results_df = pd.DataFrame(results_list)
        results_df["Model_Name"] = model_name
        results_df.to_csv(
            os.path.join(model_save_path, f"{model_name}_threshold_results.csv"),
            index=False,
        )

        print(
            f"Evaluation completed. Results saved to {os.path.join(model_save_path,f'{model_name}_results.csv')}"
        )


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py <model_name> <retrain>")
        sys.exit(1)
    print("sys.argv", sys.argv)
    model_name_arg = sys.argv[1]
    retrain_flag_arg = sys.argv[2].lower() == "true"
    gpu_id = int(sys.argv[3])

    main(model_name_arg, retrain_flag_arg, gpu_id)
