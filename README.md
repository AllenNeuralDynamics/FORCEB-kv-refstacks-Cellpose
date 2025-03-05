# FORCEB-kv-refstacks-Cellpose

This capsule tests various cellpose models out of the box and the best model is then used to train to detect cells from voltage imaging data. 

## **Data Asset Overview**
- CO Data Asset: `3c6c4e11-2852-4c9d-9051-19efb500f585`
- CO Capsule: `https://codeocean.allenneuraldynamics.org/capsule/3613171/tree`

This dataset contains two types of files:
- Files ending with `_merged.tif`: These are the actual data files.
- Files ending with `_segmented_v2.tif`: These are the corresponding mask files.

The dataset has been structured to facilitate easy interpretation and usage for downstream tasks, such as image analysis or machine learning.

---

## **Dataset Structure**
The dataset is organized as follows:

```
Root Directory/
│
├── Data/
│   ├── sample1_merged.tif
│   ├── sample1_segmented_v2.tif
│   ├── sample2_merged.tif
│   ├── sample2_segmented_v2.tif
│   └── ...
```

---
## **System requirements**
You may want to adjust the `GPU_COUNT` based on how many GPU you have available and `MODEL_NAMES` as well to the processes in parallel. 

## **Image Channel info**
Channel 1 (red) was used and channel 0 (green) was discarded from dataset as channel 1 had more detailed outline of the cells. 
<img width="1416" alt="Channel_Info" src="https://github.com/user-attachments/assets/7eaf7886-dd08-4848-a695-6a6da56878d6" />



# Testing Model out of the box:
Go to [run.sh](code/run) and change `RETRAIN=false` and adjust the list from `MODEL_NAMES` to your liking. It would test cellpose and save the results to results folders which would include model_name_results.csv and images of masks generated. 

## Example of out the box performance:
<img width="1418" alt="Cellpose_OutBoxPerf" src="https://github.com/user-attachments/assets/705d2ad8-46e1-4d9c-966e-feffbd735807" />


# Training Model out of the box:
Go to [run.sh](code/run) and change `RETRAIN=true` and adjust the list from `MODEL_NAMES` to your liking. It would test cellpose and save the results to results folders in CO. 

## Example of performance after retraining the model:
<img width="1415" alt="Screenshot 2025-01-23 at 10 32 57 AM" src="https://github.com/user-attachments/assets/c5492356-d115-40cb-9fe2-6936a22f2738" />

