import numpy as np
import os
import sys
from PIL import Image

def main():
    # 0. 参数解析
    target_idx = 4 # 默认值
    if len(sys.argv) > 1:
        target_idx = int(sys.argv[1])
        
    print(f"=== 验证与恢复: 图片索引 {target_idx} ===")

    # 配置参数
    base_dir = "dataset_downloaded/mnist_float"
    num_samples = 5000
    num_classes = 10
    label_file = os.path.join(base_dir, "label_onehot")
    img_path = os.path.join(base_dir, f"{target_idx}.bin")

    # 1. 检查文件是否存在
    if not os.path.exists(img_path):
        print(f"[Error] 图片文件不存在: {img_path}")
        return
    if not os.path.exists(label_file):
        print(f"[Error] 标签文件不存在: {label_file}")
        return

    # 2. 读取图片
    img_float = np.fromfile(img_path, dtype=np.float32)
    if img_float.size != 784:
        print(f"[Error] 图片尺寸错误: {img_float.size}")
        return

    # 3. 读取标签
    # 计算偏移量: target_idx * 10 * 4 bytes
    offset = target_idx * 10 * 4
    with open(label_file, "rb") as f:
        f.seek(offset)
        label_raw = np.frombuffer(f.read(10 * 4), dtype=np.float32)
    
    digit = np.argmax(label_raw)
    print(f"[Info] 标签识别为数字: {digit}")
    print(f"[Info] One-hot 向量: {label_raw}")

    # 4. 生成 PNG
    print(f"\n=== 生成 PNG 用于比对 ===")
    img_uint8 = (img_float * 255).astype(np.uint8)
    img_reshaped = img_uint8.reshape(28, 28)
    
    output_png_path = f"image_{target_idx}_label_{digit}.png"
    try:
        image = Image.fromarray(img_reshaped, mode='L')
        image.save(output_png_path)
        print(f"[Success] 图片已保存为: {output_png_path}")
        print(f"       请打开此图片，确认它是否与推理结果 'Prediction' 一致。")
    except Exception as e:
        print(f"[Error] 生成图片失败: {e}")

if __name__ == "__main__":
    main()