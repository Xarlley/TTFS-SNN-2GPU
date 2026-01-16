import numpy as np
import os
from PIL import Image
import tensorflow as tf

def main():
    # 1. 确定文件路径（复用之前下载好的文件）
    base_dir = "dataset_downloaded"
    # 注意：tf.keras.utils.get_file 默认下载的文件名为 mnist.npz
    file_path = os.path.join(base_dir, "mnist", "mnist.npz")

    # 如果找不到文件，再次调用 API 确保下载/定位
    if not os.path.exists(file_path):
        print(f"[Info] 在 {file_path} 未找到文件，正在尝试定位...")
        file_path = tf.keras.utils.get_file(
            'mnist.npz',
            origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz',
            cache_dir=base_dir,
            cache_subdir='mnist'
        )

    print(f"[Info] 加载原始数据集: {file_path}")

    # 2. 加载数据
    with np.load(file_path) as data:
        x_train = data['x_train']
        y_train = data['y_train']

    # 3. 定位到对应的图片
    # 逻辑：验证集是 x_train 的最后 5000 张
    # 所以验证集的 index 0 对应 x_train 的 index 55000
    original_index = 55000 
    
    print(f"[Info] 提取原始训练集索引: {original_index}")
    
    # 获取图片数据 (28x28, uint8, 0-255)
    original_img_data = x_train[original_index]
    original_label = y_train[original_index]

    print(f"       原始标签值: {original_label}")

    # 4. 保存为 PNG
    output_filename = "original_mnist_index_55000.png"
    img = Image.fromarray(original_img_data)
    img.save(output_filename)

    print(f"[Success] 原始图片已保存为: {output_filename}")
    print("-" * 50)
    print(f"现在你可以对比以下两张图片：")
    print(f"1. first_image_check.png (从你的二进制浮点数据还原)")
    print(f"2. {output_filename} (直接来自原始数据集)")
    print(f"如果两者看起来完全一样，说明数据处理管道是正确的。")

if __name__ == "__main__":
    main()