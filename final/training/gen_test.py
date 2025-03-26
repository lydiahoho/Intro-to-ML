import os
import shutil

def copy_images(source_folder, destination_folder):
    # 遍历源文件夹中的每个小文件夹
    for category_folder in os.listdir(source_folder):
        category_path = os.path.join(source_folder, category_folder)

        # 检查是否是文件夹
        if os.path.isdir(category_path):
            # 创建目标文件夹的路径
            destination_category_path = os.path.join(destination_folder, category_folder)
            os.makedirs(destination_category_path, exist_ok=True)

            # 获取当前类别文件夹中的文件列表
            files = os.listdir(category_path)

            # 选择前3张照片
            selected_files = files[:10]

            # 复制选定的文件到目标文件夹
            for file in selected_files:
                source_file_path = os.path.join(category_path, file)
                destination_file_path = os.path.join(destination_category_path, file)
                shutil.copyfile(source_file_path, destination_file_path)

# 输入的大文件夹路径
source_folder_path = 'dataset/train'

# 输出的目标文件夹路径
destination_folder_path = 'dataset/test_cur'

# 复制照片
copy_images(source_folder_path, destination_folder_path)
