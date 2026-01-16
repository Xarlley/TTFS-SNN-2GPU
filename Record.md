# 如何走到现在



这个是改造原有main函数，导出原项目的cnn_mnist权重。

```bash
python main_inject_getweight.py     --dataset=MNIST     --ann_model=CNN     --model_name=mnist_cnn_demo     --nn_mode=SNN     --en_train=False     --f_fused_bn=True     --f_w_norm_data=True     --f_write_stat=False     --neural_coding=TEMPORAL     --input_spike_mode=TEMPORAL     --n_type=IF     --n_init_vth=1.0     --tc=20     --time_fire_start=80     --time_fire_duration=80     --time_window=80     --time_step=400     --f_train_time_const=False     --f_load_time_const=True     --time_const_num_trained_data=60000     --f_refractory=True     --f_record_first_spike_time=True     --batch_size=100
```

这个是改造原有main函数，输出对mnist第一张图片进行TTFS-SNN推理的全部参数细节。

```bash
python main_debug_snn.py     --dataset=MNIST     --ann_model=CNN     --model_name=mnist_cnn_demo     --nn_mode=SNN     --en_train=False     --f_fused_bn=True     --f_w_norm_data=True     --f_write_stat=False     --neural_coding=TEMPORAL     --input_spike_mode=TEMPORAL     --n_type=IF     --n_init_vth=1.0     --tc=20     --time_fire_start=80     --time_fire_duration=80     --time_window=80     --time_step=400     --f_train_time_const=False     --f_load_time_const=True     --time_const_num_trained_data=60000     --f_refractory=True     --f_record_first_spike_time=True     --batch_size=100 > result.log
```

