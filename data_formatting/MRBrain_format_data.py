import os
import shutil

# ---
# IMPORTANT: PLEASE MODIFY THESE TWO PATHS
# ---
# Path to the folder containing the subject folders (e.g., '1', '2', '3'...)
SOURCE_TRAINING_DIR = r"C:\Hadar\Msc\Simster_2_Spring_25\DL\mini_project\data\grey_white_classification\MRBrainS18\training\training"

# Path where the new nnU-Net task folder will be created
OUTPUT_TASK_DIR = r"C:\Users\Hadar\PycharmProjects\DLProject\nnUnet_raw\Dataset102_GreyWhite"
# ---

# Define the names for the output directories
images_tr_dir = os.path.join(OUTPUT_TASK_DIR, "imagesTr")
labels_tr_dir = os.path.join(OUTPUT_TASK_DIR, "labelsTr")

# Create the necessary output directories if they don't exist
os.makedirs(images_tr_dir, exist_ok=True)
os.makedirs(labels_tr_dir, exist_ok=True)

print(f"Output directories created at: {OUTPUT_TASK_DIR}")

# Get a list of all subject folders in the source directory
subject_folders = [f for f in os.listdir(SOURCE_TRAINING_DIR) if os.path.isdir(os.path.join(SOURCE_TRAINING_DIR, f))]

processed_count = 0
# Loop through each subject folder
for subject_id in subject_folders:
    subject_path = os.path.join(SOURCE_TRAINING_DIR, subject_id)
    print(f"--- Processing Subject: {subject_id} ---")

    # Define the full paths to the source files
    t1_path = os.path.join(subject_path, "pre", "reg_T1.nii.gz")
    flair_path = os.path.join(subject_path, "pre", "FLAIR.nii.gz")
    seg_path = os.path.join(subject_path, "segm.nii.gz")

    # Check if all required files exist before proceeding
    if not all([os.path.exists(t1_path), os.path.exists(flair_path), os.path.exists(seg_path)]):
        print(f"Skipping Subject {subject_id}: One or more files are missing.")
        continue

    # Define the new filenames for the nnU-Net format
    # Case identifier can be anything, but using MRBRAINS_ is clear
    case_identifier = f"MRBRAINS-000{subject_id}"

    # Destination paths for the new files
    dest_t1 = os.path.join(images_tr_dir, f"{case_identifier}-000_0000.nii.gz")
    dest_flair = os.path.join(images_tr_dir, f"{case_identifier}-000_0001.nii.gz")
    dest_seg = os.path.join(labels_tr_dir, f"{case_identifier}-000.nii.gz")

    # Copy and rename the files
    shutil.copy2(t1_path, dest_t1)
    print(f"  Copied T1 to: {os.path.basename(dest_t1)}")

    shutil.copy2(flair_path, dest_flair)
    print(f"  Copied FLAIR to: {os.path.basename(dest_flair)}")

    shutil.copy2(seg_path, dest_seg)
    print(f"  Copied Seg to: {os.path.basename(dest_seg)}")

    processed_count += 1

print("\n-------------------------------------------")
print(f"✅ Organization complete. Processed {processed_count} subjects.")
print("Your data is now ready in the nnU-Net format.")
print("-------------------------------------------")