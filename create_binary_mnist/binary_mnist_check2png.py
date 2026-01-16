import numpy as np
import os
from PIL import Image

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

    # 读取标签
    labels_raw = np.fromfile(label_file, dtype=np.float32)
    expected_label_size = num_samples * num_classes
    
    if labels_raw.size != expected_label_size:
        print(f"[Error] 标签文件大小不匹配! 期望元素数: {expected_label_size}, 实际: {labels_raw.size}")
        return
    
    labels = labels_raw.reshape(num_samples, num_classes)
    print(f"[Pass] 标签文件读取成功。形状: {labels.shape}")

    # 2. 验证图片文件 (抽样)
    print("[Info] 正在抽样检查图片文件...")
    check_indices = [0, 100, 2500, 4999] 
    
    for idx in check_indices:
        img_path = os.path.join(base_dir, f"{idx}.bin")
        if not os.path.exists(img_path):
            print(f"[Error] 图片 {idx}.bin 丢失!")
            return

        img_raw = np.fromfile(img_path, dtype=np.float32)

        if img_raw.size != input_dim:
            print(f"[Error] 图片 {idx}.bin 尺寸错误! 期望: {input_dim}, 实际: {img_raw.size}")
            return
        
        if img_raw.min() < 0.0 or img_raw.max() > 1.0:
            print(f"[Error] 图片 {idx}.bin 数值范围异常! Min: {img_raw.min()}, Max: {img_raw.max()}")
            return

    print(f"[Pass] 抽样图片检查通过。")

    # ==========================================
    # 3. 视觉验证 (新增功能)
    # ==========================================
    print("\n=== 正在生成第一张图片的 PNG 用于人工检阅 ===")
    
    # 读取第一张图 (0.bin)
    first_img_idx = 0
    img_path = os.path.join(base_dir, f"{first_img_idx}.bin")
    
    # 1. 读取原始浮点数据 (0.0 - 1.0)
    img_float = np.fromfile(img_path, dtype=np.float32)
    
    # 2. 还原为亮暗值 (0 - 255)
    # 乘以 255 并转换为无符号 8 位整数
    img_uint8 = (img_float * 255).astype(np.uint8)
    
    # 3. 重塑为 2D 图片形状 (28x28)
    img_reshaped = img_uint8.reshape(28, 28)
    
    # 4. 保存为 PNG
    output_png_path = "first_image_check.png"
    try:
        image = Image.fromarray(img_reshaped, mode='L') # 'L' 表示灰度模式
        image.save(output_png_path)
        print(f"[Success] 图片已生成: {output_png_path}")
        print(f"       你可以打开该文件检查是否为手写数字。")
        
        # 打印对应的标签信息供对照
        label_vec = labels[first_img_idx]
        digit = np.argmax(label_vec)
        print(f"       根据标签文件，这张图应该是数字: {digit}")
        print(f"       One-hot 标签: {label_vec}")
        
    except Exception as e:
        print(f"[Error] 生成图片失败: {e}")

    # ==========================================

    # 4. 统计检查
    file_count = len([name for name in os.listdir(base_dir) if name.endswith(".bin")])
    if file_count != num_samples:
         print(f"[Error] .bin 图片文件数量不对! 期望: {num_samples}, 实际: {file_count}")
    else:
         print(f"[Pass] 目录中包含 {file_count} 个图片文件。")

    print("\n=== 验证全部完成 ===")

if __name__ == "__main__":
    main()