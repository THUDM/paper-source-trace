import os
import argparse
from datetime import datetime
import numpy as np
from sklearn.metrics import top_k_accuracy_score
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForMultipleChoice
import deepspeed
from glm.dataset import TensorDataset
from utils import Log

BATCH_SIZE = 8

def add_argument():

    parser = argparse.ArgumentParser(description='CIFAR')

    #data
    # cuda
    parser.add_argument('--with_cuda',
                        default=False,
                        action='store_true',
                        help='use CPU in case there\'s no GPU support')
    parser.add_argument('--use_ema',
                        default=False,
                        action='store_true',
                        help='whether use exponential moving average')

    # train
    parser.add_argument('-b',
                        '--batch_size',
                        default=8,
                        type=int,
                        help='mini-batch size (default: 32)')
    parser.add_argument('-e',
                        '--epochs',
                        default=50,
                        type=int,
                        help='number of total epochs (default: 30)')
    parser.add_argument('--local_rank',
                        type=int,
                        default=-1,
                        help='local rank passed from distributed launcher')

    parser.add_argument('--log-interval',
                        type=int,
                        default=2000,
                        help="output logging information at a given interval")

    parser.add_argument('--moe',
                        default=False,
                        action='store_true',
                        help='use deepspeed mixture of experts (moe)')

    parser.add_argument('--ep-world-size',
                        default=1,
                        type=int,
                        help='(moe) expert parallel world size')
    parser.add_argument('--num-experts',
                        type=int,
                        nargs='+',
                        default=[
                            1,
                        ],
                        help='number of experts list, MoE related.')
    parser.add_argument(
        '--mlp-type',
        type=str,
        default='standard',
        help=
        'Only applicable when num-experts > 1, accepts [standard, residual]')
    parser.add_argument('--top-k',
                        default=1,
                        type=int,
                        help='(moe) gating top 1 and 2 supported')
    parser.add_argument(
        '--min-capacity',
        default=0,
        type=int,
        help=
        '(moe) minimum capacity of an expert regardless of the capacity_factor'
    )
    parser.add_argument(
        '--noisy-gate-policy',
        default=None,
        type=str,
        help=
        '(moe) noisy gating (only supported with top-1). Valid values are None, RSample, and Jitter'
    )
    parser.add_argument(
        '--moe-param-group',
        default=False,
        action='store_true',
        help=
        '(moe) create separate moe param groups, required when using ZeRO w. MoE'
    )

    # Include DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()

    return args


def glm_get_params_for_weight_decay_optimization(module):
    weight_decay_params = {'params': []}
    no_weight_decay_params = {'params': [], 'weight_decay': 0.0}
    for module_ in module.modules():
        if isinstance(module_, (torch.nn.LayerNorm)):
            no_weight_decay_params['params'].extend(
                [p for p in list(module_._parameters.values())
                 if p is not None and p.requires_grad])
        else:
            weight_decay_params['params'].extend(
                [p for n, p in list(module_._parameters.items())
                 if p is not None and p.requires_grad and n != 'bias'])
            no_weight_decay_params['params'].extend(
                [p for n, p in list(module_._parameters.items())
                 if p is not None and p.requires_grad and n == 'bias'])

    return weight_decay_params, 


def get_optimizer_param_groups(model):
    # Build parameter groups (weight decay and non-decay).
    # while isinstance(model, (LocalDDP, TorchDDP, FP16_Module)):
    #     model = model.module
    param_groups = glm_get_params_for_weight_decay_optimization(model)

    # Add model parallel attribute if it is not set.
    for param_group in param_groups:
        # print('## param_group', len(param_group['params']))
        for param in param_group['params']:
            if not hasattr(param, 'model_parallel'):
                param.model_parallel = False

    return param_groups


def train(model_engine, dataloader, tokenizer, fp16):
    loss_total = 0
    n_item = 0
    for data in dataloader:
        # print("here3")
        # data = {k: v.to(model_engine.local_rank) for k, v in batch.items()}
        # if fp16:
        #     data = {k: v.half() for k, v in data.items()}
        # outputs = model_engine(**data)
        # loss = outputs.loss
        # model_engine.backward(loss)
        # model_engine.step()  

        # loss_total += loss.item()
        # cur_n_item = outputs.logits.shape[0]
        # n_item += cur_n_item
        inputs = tokenizer(data["contexts"], return_tensors="pt", padding=True)
        inputs = tokenizer.build_inputs_for_generation(inputs, targets=data["answers"], max_gen_length=2, padding=False)
        inputs = inputs.to(model_engine.local_rank)

        outputs = model_engine(**inputs)
        loss = outputs.loss
        model_engine.backward(loss)
        model_engine.step()  

        loss_total += loss.item()
        cur_n_item = outputs.logits.shape[0]
        n_item += cur_n_item
    
    return loss_total / n_item


def eval(dataloader, model_infer_ds, tokenizer, fp16):
    # model_infer = AutoModelForMultipleChoice.from_pretrained(model_name, trust_remote_code=True)
    # model_infer.load_state_dict(model.state_dict())
    # model_infer = model_infer.half()
    # model_infer = model_infer.half().cuda()  # half?
    # model_infer.eval()

    # ds_engine = deepspeed.init_inference(model_infer, mp_size=1, dtype=torch.half, replace_with_kernel_inject=True)
    # model_ds = ds_engine.module

    pred_scores = np.empty((0, 6))
    labels = []

    with torch.no_grad():
        for data in dataloader:
            label = data["labels"]
            inputs = tokenizer(data["contexts"], return_tensors="pt", padding=True)
            inputs = tokenizer.build_inputs_for_multiple_choice(inputs, data["candidates"])
            inputs = inputs.to('cuda')
            # if fp16:
            #     inputs = {k: v.half() if torch.is_tensor(v) else v for k, v in inputs.items()}
            # outputs = model_infer(**inputs)
            outputs = model_infer_ds(**inputs)
            logits = outputs.logits
            score = logits.detach().cpu().numpy()
            if score.shape[0] > 6:
                score = score[:6]
            elif score.shape[0] < 6:
                score = np.concatenate((score, np.zeros((6 - score.shape[0], score.shape[1]))), axis=0)
            pred_scores = np.concatenate((pred_scores, np.transpose(score)), axis=0)
            labels.extend(label)

    hit_1 = top_k_accuracy_score(labels, pred_scores, k=1)
    hit_3 = top_k_accuracy_score(labels, pred_scores, k=3)
    hit_5 = top_k_accuracy_score(labels, pred_scores, k=5)

    # del model_infer

    return [hit_1, hit_3, hit_5]


def finetune_ds(model_name="THUDM/glm-2b"):
    deepspeed.init_distributed()

    # if torch.distributed.get_rank() != 0:
    #     # might be downloading data, let rank 0 download first
    #     torch.distributed.barrier()

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True)
    # tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    # model = AutoModelForSeq2SeqLM.from_pretrained(model_name, local_files_only=True)
    model = model.half()

    model_infer = AutoModelForMultipleChoice.from_pretrained(model_name, trust_remote_code=True)
    model_infer = model_infer.half()

    args = add_argument()

    exp_name = "finetune-" + model_name.split('/')[-1] + '-ds-'
    exp_name = exp_name + str(datetime.now())

    # Dataset
    train_dataset_path = "glm/data/train.json"
    valid_dataset_path = "glm/data/valid.json"
    test_dataset_path = "glm/data/test.json"
    log_path = "./log/" + exp_name + ".log"
    os.makedirs("./log", exist_ok=True)

    train_dataset = TensorDataset(train_dataset_path, tokenizer)
    valid_dataset = TensorDataset(valid_dataset_path, tokenizer)
    test_dataset = TensorDataset(test_dataset_path, tokenizer)

    # Log
    log = Log(file_path= log_path)

    # initialize the dataloader
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    parameters = get_optimizer_param_groups(model)

    print("here1")
    model_engine, optimizer, trainloader, __ = deepspeed.initialize(
    args=args, model=model, model_parameters=parameters, training_data=train_dataset)
    print("here2")

    ds_engine_infer = deepspeed.init_inference(model_infer, mp_size=1, dtype=torch.half, replace_with_kernel_inject=True)

    fp16 = model_engine.fp16_enabled()
    print(f'fp16={fp16}')   

    # ds_engine_infer._load_from_state_dict(model.state_dict())
    ds_engine_infer.module.load_state_dict(model.state_dict())
    # model_infer_ds = ds_engine_infer.module
    # res_valid = eval(valid_loader, model_infer_ds, tokenizer, fp16)
    # log.log('valid epoch {} result: {}'.format(-1, str(res_valid)))
    # res_test = eval(test_loader, model_infer_ds, tokenizer, fp16)
    # log.log('test epoch {} result: {}'.format(-1, str(res_test)))   

    valid_best = 0
    test_best = []
    best_epoch = 0
    out_dir = "glm/saved"
    os.makedirs(out_dir, exist_ok=True)
    for epoch in range(args.epochs):
        log.log('begin epoch {}'.format(epoch))
        loss_train = train(model_engine, trainloader, tokenizer, fp16)
        log.log('train epoch {} end, loss {}'.format(epoch, str(loss_train)))

        # ds_engine_infer._load_from_state_dict(model.state_dict())

        # ds_engine_infer.module.load_state_dict(model.state_dict())
        # model_infer_ds = ds_engine_infer.module
        # res_valid = eval(valid_loader, model_infer_ds, tokenizer, fp16)
        # log.log('valid epoch {} result: {}'.format(epoch, str(res_valid)))
        # res_test = eval(test_loader, model_infer_ds, tokenizer, fp16)
        # log.log('test epoch {} result: {}'.format(epoch, str(res_test)))     

        # if res_valid[0] > valid_best:
        #     valid_best = res_valid[0]
        #     test_best = res_test
        #     best_epoch = epoch
        # log.log('best epoch {} result: {}'.format(best_epoch, str(test_best)))

        ds_engine_infer.module.load_state_dict(model.state_dict())
        torch.save(ds_engine_infer.module, os.path.join(out_dir, "{}-epoch-{}.pt".format("glm-10b", epoch)))



if __name__ == "__main__":
    model_name = "THUDM/glm-10b"
    finetune_ds(model_name)
