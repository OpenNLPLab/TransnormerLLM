GPUs=$1

torchrun \
    --nproc_per_node=$GPUs \
    train.py \
    --model_name_or_path /cpfs01/shared/MMG/llm-7b/checkpoints/eval_ckpt_hf/7B-190837 \
    --data_path ./data/alpaca_data.json \
    --output_dir output/test \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --bf16 true \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 5000 \
    --save_total_limit 30 \
    --learning_rate 1e-4 \
    --weight_decay 0.1 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --deepspeed 'configs/zero3.json' \
    --logging_steps 1 \
    --dataloader_num_workers 24 \
    --ddp_find_unused_parameters false \
    --tf32 true \
