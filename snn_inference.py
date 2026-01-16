import torch
import torch.nn as nn
import numpy as np
import struct
import os
import sys

# ==========================================
# 0. 全局配置 (必须与 snn_inference.cu 一致)
# ==========================================
TIME_WINDOW = 80.0  # 必须是 80
DT = 1.0            # 时间步长

# ==========================================
# 1. 权重加载工具
# ==========================================
def load_exported_weights(filename):
    print(f"Loading weights from {filename}...")
    layers_data = {}
    if not os.path.exists(filename):
        print(f"Error: {filename} not found.")
        sys.exit(1)
        
    with open(filename, 'rb') as f:
        # 读取层数
        try:
            num_layers_bytes = f.read(4)
            if not num_layers_bytes: raise ValueError("Empty file")
            num_layers = struct.unpack('i', num_layers_bytes)[0]
        except:
            print("Error reading file header.")
            sys.exit(1)

        layer_names = ['conv1', 'conv2', 'fc1']
        
        for name in layer_names:
            # 1. Kernel
            ndim = struct.unpack('i', f.read(4))[0]
            dims = struct.unpack(f'{ndim}i', f.read(4 * ndim))
            k_data = np.frombuffer(f.read(4 * np.prod(dims)), dtype=np.float32).reshape(dims)
            
            # 2. Bias
            ndim = struct.unpack('i', f.read(4))[0]
            dims = struct.unpack(f'{ndim}i', f.read(4 * ndim))
            b_data = np.frombuffer(f.read(4 * np.prod(dims)), dtype=np.float32)
            
            # 3. TC
            ndim = struct.unpack('i', f.read(4))[0]
            dims = struct.unpack(f'{ndim}i', f.read(4 * ndim))
            tc_data = np.frombuffer(f.read(4 * np.prod(dims)), dtype=np.float32)
            
            # 4. TD
            ndim = struct.unpack('i', f.read(4))[0]
            dims = struct.unpack(f'{ndim}i', f.read(4 * ndim))
            td_data = np.frombuffer(f.read(4 * np.prod(dims)), dtype=np.float32)
            
            layers_data[name] = {
                'weight': k_data, 
                'bias': b_data, 
                'tc': tc_data[0], 
                'td': td_data[0]
            }
            # print(f"Loaded {name}: TC={tc_data[0]:.2f}, TD={td_data[0]:.2f}")
    return layers_data

# ==========================================
# 2. TTFS 神经元 & 状态池化
# ==========================================

class TemporalNeuron(nn.Module):
    def __init__(self, tc, td, v_init=1.0):
        super().__init__()
        self.tc = tc
        self.td = td
        self.v_init = v_init
        self.register_buffer('v_mem', torch.zeros(1))
        self.register_buffer('has_fired', torch.zeros(1))
    
    def reset_state(self, batch_size, spatial_shape, device):
        shape = (batch_size, *spatial_shape)
        self.v_mem = torch.zeros(shape, device=device)
        self.has_fired = torch.zeros(shape, dtype=torch.bool, device=device)
    
    def forward(self, input_current, t):
        # 积分: 累加输入 (Bias 包含在 input_current 中)
        # 对应 CUDA: v_mem += weight + bias
        self.v_mem = self.v_mem + input_current
        
        # 动态阈值
        t_rel = t - self.td
        v_th = self.v_init * torch.exp(-t_rel / self.tc)
        
        # 发放判定
        fire_mask = (self.v_mem >= v_th) & (~self.has_fired) & (v_th > 1e-5)
        self.has_fired = self.has_fired | fire_mask
        
        return fire_mask.float()

class TTFSPooling(nn.Module):
    """
    模拟 CUDA 的 'Min-Time Pooling'。
    逻辑：在 2x2 窗口内，只允许最早的一个脉冲通过。
    """
    def __init__(self, kernel_size=2, stride=2):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.register_buffer('fired_mask', torch.zeros(1))
        
    def reset_state(self, batch_size, channels, height, width, device):
        h_out = height // self.stride
        w_out = width // self.stride
        self.fired_mask = torch.zeros((batch_size, channels, h_out, w_out), dtype=torch.bool, device=device)
        
    def forward(self, x):
        # 1. 找出当前时刻有脉冲的窗口 (Standard MaxPool behaves like OR for binary spikes)
        current_pool = torch.nn.functional.max_pool2d(x, self.kernel_size, self.stride)
        
        # 2. 过滤掉已经发放过的窗口 (TTFS Logic)
        valid_spikes = (current_pool > 0.5) & (~self.fired_mask)
        
        # 3. 更新状态
        self.fired_mask = self.fired_mask | valid_spikes
        
        return valid_spikes.float()

# ==========================================
# 3. 完整的 SNN 网络
# ==========================================
class MnistSNN_Exact(nn.Module):
    def __init__(self, weights_data):
        super().__init__()
        
        # --- Layer 1: Conv1 ---
        # TF(H,W,I,O) -> PT(O,I,H,W)
        w1 = torch.from_numpy(weights_data['conv1']['weight']).permute(3, 2, 0, 1)
        b1 = torch.from_numpy(weights_data['conv1']['bias'])
        self.conv1 = nn.Conv2d(1, 12, kernel_size=5, stride=1, padding=0)
        self.conv1.weight.data = w1
        self.conv1.bias.data = b1
        self.n1 = TemporalNeuron(weights_data['conv1']['tc'], weights_data['conv1']['td'])
        self.pool1 = TTFSPooling(2, 2)
        
        # --- Layer 2: Conv2 ---
        w2 = torch.from_numpy(weights_data['conv2']['weight']).permute(3, 2, 0, 1)
        b2 = torch.from_numpy(weights_data['conv2']['bias'])
        self.conv2 = nn.Conv2d(12, 64, kernel_size=5, stride=1, padding=0)
        self.conv2.weight.data = w2
        self.conv2.bias.data = b2
        self.n2 = TemporalNeuron(weights_data['conv2']['tc'], weights_data['conv2']['td'])
        self.pool2 = TTFSPooling(2, 2)
        
        # --- Layer 3: FC1 ---
        # TF(I,O) -> PT(O,I)
        w3 = torch.from_numpy(weights_data['fc1']['weight']).t() 
        b3 = torch.from_numpy(weights_data['fc1']['bias'])
        self.fc1 = nn.Linear(1024, 10)
        self.fc1.weight.data = w3
        self.fc1.bias.data = b3
        self.n3 = TemporalNeuron(weights_data['fc1']['tc'], weights_data['fc1']['td'])

    def forward(self, input_image):
        device = input_image.device
        batch_size = input_image.shape[0]
        
        # --- Input Encoding (Strict Match with Main Log) ---
        # 从 main_inject.py 日志中获取的真实参数
        tc_in = 17.452274
        td_in = 0.0
        
        # 1. 像素预处理 (匹配 CUDA)
        pixel = input_image.clone()
        # 如果是 0-255，归一化 (虽然日志显示是 0-1，但加保险无害)
        if pixel.max() > 1.05:
            pixel = pixel / 255.0
            
        pixel = torch.clamp(pixel, min=1e-5) # 基础保护
        
        # 2. TTFS 编码公式
        spike_times_in = td_in - tc_in * torch.log(pixel)
        spike_times_in = torch.clamp(spike_times_in, min=0.0)
        spike_times_in = torch.ceil(spike_times_in)
        
        # --- Reset States ---
        self.n1.reset_state(batch_size, (12, 24, 24), device)
        self.pool1.reset_state(batch_size, 12, 24, 24, device)
        
        self.n2.reset_state(batch_size, (64, 8, 8), device)
        self.pool2.reset_state(batch_size, 64, 8, 8, device)
        
        self.n3.reset_state(batch_size, (10,), device)
        
        out_spike_times = torch.full((batch_size, 10), 9999.0, device=device)
        
        # --- Simulation Loop (0 to 80) ---
        for t in range(int(TIME_WINDOW)):
            t_tensor = torch.tensor(t, device=device).float()
            
            # 1. Input Spikes
            mask_in = (torch.abs(t_tensor - spike_times_in) < 0.5).float()
            
            # 2. Conv1
            curr1 = self.conv1(mask_in)
            spk1 = self.n1(curr1, t_tensor)
            spk1_p = self.pool1(spk1)
            
            # 3. Conv2
            curr2 = self.conv2(spk1_p)
            spk2 = self.n2(curr2, t_tensor)
            spk2_p = self.pool2(spk2)
            
            # 4. FC1 (Critical: Channels Last Flatten)
            # PyTorch: (B, C, H, W) -> Permute(0,2,3,1) -> (B, H, W, C) -> Flatten
            flat_in = spk2_p.permute(0, 2, 3, 1).contiguous().view(batch_size, -1)
            
            curr3 = self.fc1(flat_in)
            spk3 = self.n3(curr3, t_tensor)
            
            # Record
            fired_idx = torch.nonzero(spk3, as_tuple=True)
            for b, c in zip(*fired_idx):
                if out_spike_times[b, c] == 9999.0:
                    out_spike_times[b, c] = t_tensor
        
        return out_spike_times

def main():
    target_idx = 4
    if len(sys.argv) > 1: target_idx = int(sys.argv[1])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running STRICT PyTorch Verification on {device} (Image {target_idx})")
    
    # 1. Load Image (Binary)
    base_dir = "dataset_downloaded/mnist_float"
    img_path = os.path.join(base_dir, f"{target_idx}.bin")
    if not os.path.exists(img_path):
        print(f"Image not found: {img_path}")
        return
        
    img_raw = np.fromfile(img_path, dtype=np.float32).reshape(1, 28, 28)
    print(f"Image Stats: Min={img_raw.min():.4f}, Max={img_raw.max():.4f}, Mean={img_raw.mean():.4f}")
    
    # 2. Load Weights (Hijacked)
    weights_path = "exported_models/snn_weights.bin"
    if not os.path.exists(weights_path):
        print(f"Weights not found: {weights_path}")
        return
    weights = load_exported_weights(weights_path)
    
    # 3. Run Inference
    net = MnistSNN_Exact(weights).to(device)
    img_tensor = torch.from_numpy(img_raw).unsqueeze(0).to(device)
    
    with torch.no_grad():
        out_times = net(img_tensor)
        
    times = out_times.cpu().numpy()[0]
    
    # 4. Report
    print(f"\n=== PyTorch Result (Window={TIME_WINDOW}) ===")
    min_t = 9999.0
    pred_cls = -1
    
    for i in range(10):
        t = times[i]
        if t < 9999:
            print(f"Class {i}: {t:.2f}")
            if t < min_t:
                min_t = t
                pred_cls = i
        else:
            print(f"Class {i}: Did not fire")
            
    print("--------------------------------")
    print(f"Prediction: {pred_cls}")

if __name__ == "__main__":
    main()