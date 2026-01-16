import numpy as np
import os
import struct

def main():
    # 配置参数
    base_dir = "dataset_downloaded/mnist_float"
    num_samples = 5000
    input_dim = 784  # 28*28
    num_classes = 10
    label_file = os.path.join(base_dir, "label_onehot")

    print("=== 开始验证数据完整性 ===")

    # 1. 验证标签文件
    if not os.path.exists(label_file):
        print(f"[Error] 找不到标签文件: {label_file}")
        return

    # 模拟 C++ 读取：直接读取 float32 数组
    # 对应 C++: fread(buffer, sizeof(float), 5000 * 10, file);
    labels_raw = np.fromfile(label_file, dtype=np.float32)
    
    # 检查大小
    expected_label_size = num_samples * num_classes
    if labels_raw.size != expected_label_size:
        print(f"[Error] 标签文件大小不匹配! 期望元素数: {expected_label_size}, 实际: {labels_raw.size}")
        return
    
    # 重塑形状以方便检查
    labels = labels_raw.reshape(num_samples, num_classes)
    print(f"[Pass] 标签文件读取成功。形状: {labels.shape}")
    print(f"       第一张标签样例: {labels[0]}")

    # 2. 验证图片文件
    print("[Info] 正在抽样检查图片文件...")
    
    # 随机抽取 5 张检查，或者检查全部
    check_indices = [0, 100, 2500, 4999] 
    
    for idx in check_indices:
        img_path = os.path.join(base_dir, f"{idx}.bin")
        
        if not os.path.exists(img_path):
            print(f"[Error] 图片 {idx}.bin 丢失!")
            return

        # 模拟 C++ 读取：直接读取 float32 数组
        # 对应 C++: fread(buffer, sizeof(float), 784, file);
        img_raw = np.fromfile(img_path, dtype=np.float32)

        # 检查尺寸
        if img_raw.size != input_dim:
            print(f"[Error] 图片 {idx}.bin 尺寸错误! 期望: {input_dim}, 实际: {img_raw.size}")
            return
        
        # 检查数值范围 (应该在 0.0 到 1.0 之间)
        if img_raw.min() < 0.0 or img_raw.max() > 1.0:
            print(f"[Error] 图片 {idx}.bin 数值范围异常! Min: {img_raw.min()}, Max: {img_raw.max()}")
            return

    print(f"[Pass] 抽样图片检查通过。")
    print(f"       图片 {check_indices[0]}.bin 前10个像素值: {img_raw[:10]}") # 这里的img_raw是循环中最后一张，仅作打印示例

    # 3. 统计检查
    file_count = len([name for name in os.listdir(base_dir) if name.endswith(".bin")])
    if file_count != num_samples:
         print(f"[Error] .bin 图片文件数量不对! 期望: {num_samples}, 实际: {file_count}")
    else:
         print(f"[Pass] 目录中包含 {file_count} 个图片文件。")

    print("\n=== 验证成功：数据格式符合 C++ 读取要求 ===")

if __name__ == "__main__":
    main()