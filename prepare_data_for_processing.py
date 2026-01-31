import os
import nibabel as nib

masterdata_dir = "/mnt/data/ADNI/T1"

list_img_folders = os.listdir(masterdata_dir)

for img_folder in list_img_folders:
    img_folder_path = os.path.join(masterdata_dir, img_folder)
    cat12_path = os.path.join(img_folder_path, "cat12")
    
    #Locate *nii.gz image on img_folder_path
    t1_files = [f for f in os.listdir(img_folder_path) if f.startswith("t1_") and f.endswith(".nii.gz")]
    if t1_files:
        t1_filename = t1_files[0]
        t1_name = t1_filename.replace(".nii.gz", "")
        cat12_mri_path = os.path.join(cat12_path, f"mwp1{t1_name}.nii")
        cat12_report_path = os.path.join(cat12_path, f"cat_{t1_name}.xml")

        if os.path.exists(cat12_mri_path) and os.path.exists(cat12_report_path):
            continue

        else:
            os.makedirs(cat12_path, exist_ok=True)
            img = nib.load(os.path.join(img_folder_path, t1_filename))
            nib.save(img, os.path.join(cat12_path, t1_filename.replace(".gz", "")))

    else:
        print(f"No t1_*.nii.gz file found in {img_folder_path}")
    