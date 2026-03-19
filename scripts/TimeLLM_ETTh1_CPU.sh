#!/bin/bash

# --- 专为 Intel Ultra 5 135U / 16GB 内存 定制的 CPU 训练脚本 ---
# 这份脚本取消了多 GPU 的 accelerate launch 启动方式，
# 并且直接调用轻量级的 GPT-2 模型，将 batch size 减半。

model_name=TimeLLM
train_epochs=10
learning_rate=0.01

# 使用 6 层 GPT-2 的隐藏层
llama_layers=6

# 减小 Batch Size，降低内存占用，提升 CPU 每次迭代的速度
batch_size=8

# GPT-2 的默认维度是 768
d_llm=768

# 减小模型中间层维度，加速计算
d_model=16
d_ff=32

comment='TimeLLM-ETTh1-CPU-GPT2'

echo "=========================================================="
echo "    🚀 开始使用 CPU (GPT-2) 训练 Time-LLM 模型 🚀"
echo "    当前运行的预测长度：96"
echo "=========================================================="

python run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_512_96 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 512 \
  --label_len 48 \
  --pred_len 96 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_model GPT2 \
  --llm_dim $d_llm \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --model_comment $comment \
  --patience 3

echo "🎉 训练完成！结果已保存在 ./checkpoints 目录下"