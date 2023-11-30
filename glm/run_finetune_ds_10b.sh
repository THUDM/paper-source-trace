deepspeed --master_port 23620 --include localhost:0,1,6,7 finetune_glm_ds_10b.py --deepspeed --deepspeed_config ds_config_glm_10b.json $@
