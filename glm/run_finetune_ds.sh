#!/bin/bash

deepspeed --master_port 23619 glm/finetune_glm_ds.py --deepspeed --deepspeed_config glm/ds_config_glm_2b.json $@
