accelerate launch train.py \
  --model_name_or_path "pruned_models/PruneSLU-15M-Pruned" \
  --teacher_model_name_or_path "pruned_models/PruneSLU-15M-Teacher" \
  --train_data "./data/train_data.json" \
  --eval_data "./data/eval_data.json" \
  --eval_steps 5000 \
  --save_steps 5000 \
  --warmup_steps 5000 \
  --learning_rate 0.00004 \
  --logging_steps 100 \
  --save_total_limit 5 \
  --num_train_epochs 10 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --dataloader_num_workers 2 \
  --gradient_accumulation_steps 1 \
  --ddp_timeout 7200 \
  --dtype "float16" \
  --attn_implementation "sdpa" \
  --output_dir "./outputs/PruneSLU-15M" \
  --do_train \
  --do_eval \
  --gradient_checkpointing \
  --overwrite_output_dir \
  --report_to "none" \
  --seed 0 \
  --layer_kl_weight 0.1 \
  --decoder_layers_numbers 2 3

python remove_suppress_tokens.py --base_folder outputs/PruneSLU-15M

python inference_beam_search.py --model_path=outputs/PruneSLU-15M --test_data data/test_data.json --num_beam 10