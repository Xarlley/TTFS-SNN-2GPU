import tensorflow as tf
import numpy as np
import os
import shutil

def main():
    # 1. 定义路径
    base_dir = "dataset_downloaded"
    mnist_source_dir = os.path.join(base_dir, "mnist")
    output_dir = os.path.join(base_dir, "mnist_float")
    
    # 确保目录存在
    os.makedirs(mnist_source_dir, exist_ok=True)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir) # 清空旧数据以防混淆
    os.makedirs(output_dir, exist_ok=True)

    print(f"[Info] 准备下载/加载数据到: {mnist_source_dir}")

    # 2. 下载数据到指定目录
    # 我们使用 get_file 手动下载 npz 文件，以确保它出现在你要求的文件夹中
    path = tf.keras.utils.get_file(
        'mnist.npz',
        origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz',
        cache_dir=base_dir,      # cache_dir + cache_subdir = 最终路径
        cache_subdir='mnist'     # 最终文件会在 dataset_downloaded/mnist/mnist.npz
    )

    # 3. 加载并切分数据
    print("[Info] 加载并处理数据...")
    with np.load(path) as data:
        # 加载训练集
        x_train, y_train = data['x_train'], data['y_train']
        
        # 抽取后 5000 张作为验证集
        x_val = x_train[-5000:]
        y_val = y_train[-5000:]

    # 4. 数据转换 (关键步骤)
    # 转换为 float32 并归一化到 [0, 1]
    # Flatten: 将 (5000, 28, 28) -> (5000, 784)，方便C++直接读取为一维数组
    x_val_float = x_val.reshape(x_val.shape[0], -1).astype(np.float32) / 255.0
    
    # 标签转换为 One-hot, float32 格式
    y_val_onehot = tf.one_hot(y_val, depth=10).numpy().astype(np.float32)

    print(f"[Info] 验证集形状: {x_val_float.shape}")
    print(f"[Info] 标签集形状: {y_val_onehot.shape}")

    # 5. 存储标签 (单个文件)
    # 使用 .tofile() 保存为原始二进制，C++ 可以直接 fread 读取 sizeof(float)*10*5000
    label_path = os.path.join(output_dir, "label_onehot")
    y_val_onehot.tofile(label_path)
    print(f"[Success] 标签已保存至: {label_path}")

    # 6. 存储图片 (每张图片一个文件)
    print("[Info] 正在保存 5000 张图片...")
    for i in range(x_val_float.shape[0]):
        # 文件名例如: 0.bin, 1.bin ...
        # C++ 读取时只需申请 784 * sizeof(float) 的内存
        img_path = os.path.join(output_dir, f"{i}.bin")
        x_val_float[i].tofile(img_path)
    
    print(f"[Success] 所有图片已保存至: {output_dir}")
    print("[Info] 数据处理完成。")

if __name__ == "__main__":
    main()