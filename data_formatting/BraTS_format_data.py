import os
import shutil
import gzip
import nibabel as nib
import json

# Path setup
SOURCE_DIR = "C:\Hadar\Msc\Simster_2_Spring_25\DL\mini_project\data\\tumer_classification\data_4\\try\BraTS-MEN-Train"
TARGET_BASE = "C:/Users/Hadar/PycharmProjects/DLProject/nnUnet_raw"
IMAGES_TR = os.path.join(TARGET_BASE, "imagesTr")
LABELS_TR = os.path.join(TARGET_BASE, "labelsTr")

os.makedirs(IMAGES_TR, exist_ok=True)
os.makedirs(LABELS_TR, exist_ok=True)

MODALITY_MAP = {
    "t1n": "0000",
    "t2f": "0001",
}

# Process each patient folder
for patient_folder in os.listdir(SOURCE_DIR):
    patient_path = os.path.join(SOURCE_DIR, patient_folder)
    if not os.path.isdir(patient_path):
        continue

    case_id = patient_folder  # e.g., BraTS-MEN-00004-000

    for file in os.listdir(patient_path):
        if not file.endswith(".nii.gz"):
            continue

        full_path = os.path.join(patient_path, file)

        if "seg" in file:
            # Segmentation file
            label_out = os.path.join(LABELS_TR, f"{case_id}.nii.gz")
            shutil.copy(full_path, label_out)
        else:
            # Image modality
            for modality, idx in MODALITY_MAP.items():
                if modality in file:
                    image_out = os.path.join(IMAGES_TR, f"{case_id}_{idx}.nii.gz")
                    shutil.copy(full_path, image_out)

# Create dataset.json (minimal example)
dataset_json = {
    "name": "BraTS-MEN",
    "description": "BraTS MEN for nnU-Net",
    "tensorImageSize": "4D",
    "reference": "",
    "licence": "",
    "release": "1.0",
    "modality": {
        "0": "T1c",
        "1": "T1n",
        "2": "T2f",
        "3": "T2w"
    },
    "labels": {
        "0": "background",
        "1": "tumor"
    },
    "numTraining": len(os.listdir(LABELS_TR)),
    "training": [
        {
            "image": f"./imagesTr/{file.replace('.nii.gz', '')}",
            "label": f"./labelsTr/{file}"
        } for file in os.listdir(LABELS_TR)
    ],
    "test": []
}

with open(os.path.join(TARGET_BASE, "dataset.json"), "w") as f:
    json.dump(dataset_json, f, indent=4)

