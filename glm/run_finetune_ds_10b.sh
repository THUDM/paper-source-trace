deepspeed --master_port 23620 --include localhost:2,6,7 glm/finetune_glm_10b_ds.py --deepspeed --deepspeed_config glm/ds_config_glm_10b.json $@
