
LR=1e-4

MASTER_PORT=$(shuf -n 1 -i 10000-65535)

deepspeed  --include localhost:0,2,6,7 --master_port $MASTER_PORT finetune_test.py \
    --deepspeed deepspeed.json \
    --do_predict \
    --test_file data2/test.json \
    --prompt_column content \
    --response_column summary \
    --overwrite_cache \
    --model_name_or_path /data/caokun/huge_model/chatGLM/ChatGLM-6B-main/ptuning/output/chatglm-ft/checkpoint-5000/ \
    --output_dir /data/caokun/huge_model/chatGLM/ChatGLM-6B-main/ptuning/output/test/finetune/5000/ \
    --overwrite_output_dir \
    --max_source_length 64 \
    --max_target_length 64 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --predict_with_generate \
    --max_steps 5000 \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate $LR \
    --fp16

