## Overview
This project detects and recognizes food items and tableware on a restaurant table from top-down video footage.  
Object detection is performed with **YOLOv11-small**.

## Dataset
The `data` directory follows the usual split (`train`, `val`, `test`) and includes only a few sample images.

## Training
A Jupyter notebook with the training pipeline is located in `notebooks/`.

## Inference
* `main.py` contains the main logic for running object detection on video.  
* All paths, grouped annotation lists, and other settings are configured in `config.yaml`.

# How to Reproduce

## 1  Requirements
- **Python 3.10.x**

## 2 Data Preparation

1. Extract frames from your training video(s).
2. Annotate the frames in any external tool (e.g. Roboflow).
3. Export annotations in **YOLOv11** format.
4. *(Optional)* include a **90° rotation** augmentation during export.
5. Split the dataset into the following structure:

   ```text
   data/
   ├── train/
   │   ├── images/
   │   └── labels/
   ├── val/
   │   ├── images/
   │   └── labels/
   └── test/
       ├── images/
       └── labels/
6. Add a data.yaml that lists the above paths and your class names to the data directory file. Example:

   ```yaml
   train: data/train/images
   val: data/val/images
   test: data/test/images

   nc: 4  # number of classes
   names: ['food', 'tableware', 'tableware_state', 'other']
   ```

## 3 Optional Motion-Blur Augmentation

Apply motion blur to a copy of the dataset. Set a source dataset path and an output path for the augmented dataset and
set the desired probability and blur level range.

`python augmentation/augm_motion_blur.py `
    `--source_path your_path_to_dataset `
    `--target_path your_outpur_path `
    `--probability 0.3 `
    `--blur_level_range 3 7`

Check the `augmentation/augm_motion_blur.py` script for more details on the parameters.
Make sure that the augmented and original images are merged together (if the same path is used for both --source_path and --target_path).
Otherwise, manually combine them into a single directory.
Note: This script should be applied to the training set only.

## 4 Training

Open the notebook `train_model.ipynb`.

Configure the following:
- `data_yaml_path`: path to `data.yaml`

Run the notebook `train_model.ipynb` (locally or on a cloud service such as Google Colab).
The notebook saves the best model weights to `run\\weights\\best.pt`. 

## 5 Inference & Post-Processing

Edit `config.yaml`:

- `model_path`: path to the trained model `best.pt`
- `video_path`: input video for inference
- `classes`: map your annotation classes to the internal categories  
  *(food_classes, plate_classes, plate_state_classes, other_classes)*

Launch the pipeline:

`python main.py`
main.py loads the configuration, runs detection + post-processing, and displays the annotated video stream in real time.

## 6 Validation on test set
To validate the model on the test set, run the following script:
`python final_validation.py`