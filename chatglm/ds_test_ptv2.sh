PRE_SEQ_LEN=128
LR=2e-2

CUDA_VISIBLE_DEVICES=2 python3 test.py \
    --do_predict \
    --test_file data/test.json \
    --prompt_column content \
    --response_column summary \
    --overwrite_cache \
    --model_name_or_path /data/caokun/huge_model/chatGLM/model/ \
    --output_dir output/ptuning/ \
    --overwrite_output_dir \
    --max_source_length 64 \
    --max_target_length 64 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --predict_with_generate \
    --max_steps 3000 \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate $LR \
    --pre_seq_len $PRE_SEQ_LEN \
    --quantization_bit 4 \
    # --ptuning_checkpoint /data/caokun/huge_model/chatGLM/ChatGLM-6B-main/ptuning/output/v2/checkpoint-1000/

