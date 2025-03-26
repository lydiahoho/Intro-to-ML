import os
import shutil


source_folder = '/mnt/data2/lydiaho/FGVC-HERBS/all_data/data/train' 

target_folder = '/mnt/data2/lydiaho/FGVC-HERBS/all_data/data/test2'  

for root, dirs, files in os.walk(source_folder):
    for folder in dirs:
        folder_path = os.path.join(root, folder)

        if len(os.listdir(folder_path)) >= 5:
            new_folder_path = os.path.join(target_folder, folder)
            os.makedirs(new_folder_path, exist_ok=True)
            
            selected_files = os.listdir(folder_path)[:5]
            for file in selected_files:
                file_path = os.path.join(folder_path, file)
                shutil.copy(file_path, new_folder_path)
