@echo off
set HF_ENDPOINT=https://hf-mirror.com
setlocal ENABLEDELAYEDEXPANSION
chcp 65001 > nul

:: --- 专为 客服工单数据 (按日统计) / CPU 环境定制的预测脚本 (Windows 专用) ---
:: 适用数据：cleaned_merged_data.csv (date, call_volume, tickets_received, tickets_resolved)
:: 预测目标：使用过去 90 天数据，预测未来 30 天的指标。

set model_name=TimeLLM
set train_epochs=1
set learning_rate=0.01

:: 轻量级 GPT-2 模型设置
set llama_layers=6
set batch_size=8
set d_llm=768
set d_model=16
set d_ff=32

set comment=TimeLLM-CustomerService-30Days-CPU-FastTest

echo ==========================================================
echo     📊 开始预测 客服工单数据 (未来 30 天) 📊
echo     请确保数据集位于: .\dataset\customer_service\cleaned_merged_data.csv
echo ==========================================================

python run_main.py ^
  --task_name long_term_forecast ^
  --is_training 1 ^
  --root_path .\dataset\customer_service\ ^
  --data_path cleaned_merged_data.csv ^
  --model_id CustomerService_90_30 ^
  --model %model_name% ^
  --num_workers 0 ^
  --data Custom ^
  --features M ^
  --target tickets_resolved ^
  --freq d ^
  --seq_len 90 ^
  --label_len 30 ^
  --pred_len 30 ^
  --factor 3 ^
  --enc_in 3 ^
  --dec_in 3 ^
  --c_out 3 ^
  --des Exp ^
  --itr 1 ^
  --d_model %d_model% ^
  --d_ff %d_ff% ^
  --batch_size %batch_size% ^
  --learning_rate %learning_rate% ^
  --llm_model GPT2 ^
  --llm_dim %d_llm% ^
  --llm_layers %llama_layers% ^
  --train_epochs %train_epochs% ^
  --model_comment %comment% ^
  --patience 3

echo 🎉 训练与预测完成！结果已保存在 .\checkpoints 和 .\results 目录下
pause