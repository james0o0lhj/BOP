import os
import random
import shutil

data_dir= "..\\dataset\\lmo\\test\\000002"
# 设置目录路径
label_dir = os.path.join(data_dir, "labels")
image_dir = os.path.join(data_dir, "rgb")
train_dir = os.path.join(data_dir, "train")
val_dir = os.path.join(data_dir, "val")
test_dir = os.path.join(data_dir, "test")


# 如果目录不存在，则创建目录
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# 设置拆分比例
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

# 获取标签和图像文件列表
label_files = os.listdir(label_dir)
image_files = os.listdir(image_dir)

# 随机打乱文件列表
random.shuffle(label_files)

# 计算拆分点
num_train = int(len(label_files) * train_ratio)
num_val = int(len(label_files) * val_ratio)

# 拆分文件列表
train_files = label_files[:num_train]
val_files = label_files[num_train:num_train + num_val]
test_files = label_files[num_train + num_val:]

# Copy train files
for file in train_files:
    src = os.path.join(label_dir, file)
    dst = os.path.join(train_dir, file)
    shutil.copy(src, dst)
    # Assuming corresponding images need to be copied as well
    image_src = os.path.join(image_dir, file.replace(".txt", ".png"))
    image_dst = os.path.join(train_dir, file.replace(".txt", ".png"))
    shutil.copy(image_src, image_dst)

# Copy validation files
for file in val_files:
    src = os.path.join(label_dir, file)
    dst = os.path.join(val_dir, file)
    shutil.copy(src, dst)
    # Assuming corresponding images need to be copied as well
    image_src = os.path.join(image_dir, file.replace(".txt", ".png"))
    image_dst = os.path.join(val_dir, file.replace(".txt", ".png"))
    shutil.copy(image_src, image_dst)

# Copy test files
for file in test_files:
    src = os.path.join(label_dir, file)
    dst = os.path.join(test_dir, file)
    shutil.copy(src, dst)
    # Assuming corresponding images need to be copied as well
    image_src = os.path.join(image_dir, file.replace(".txt", ".png"))
    image_dst = os.path.join(test_dir, file.replace(".txt", ".png"))
    shutil.copy(image_src, image_dst)