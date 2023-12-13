# -*- coding: utf-8 -*-

import os, sys
import datetime
import glob
import re
import pandas as pd

import argparse
import logging
import pickle
import pytz

from transformers import  ElectraConfig, ElectraTokenizer, ElectraModel
from transformers import AdamW, get_linear_schedule_with_warmup
from fastprogress.fastprogress import master_bar, progress_bar

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from misc import whoami, run_scandir, run_scandir_re
from misc import init_logger, set_seed, load_config, get_gpu

from data import get_data_from_ai_hub, get_data_from_sci_news_50, MAX_TEXT_TOKENS, MAX_SUMM_TOKENS
from data import GNN_Dataset, BatchSampler, collate_func, create_masks
from model import get_model


logger = logging.getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser(description='training of the abstractor (ML)')

    parser.add_argument('--config-dir'    ,type=str , default=os.path.join('.' 'config'), help='config. directory')
    parser.add_argument('--config-file'   ,type=str , required=True, help='config. file in config. directory')

    return parser.parse_args()


def evaluate(config, model, dataset, mode, global_step=None):
    results = dict()

    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=config.eval_batch_size)

    dataloader = DataLoader(dataset, \
                            batch_size=config.eval_batch_size, \
                            num_workers=config.n_workers_for_dataloader, \
                            collate_fn=collate_func, \
                            sampler=sampler)

    # start evaluation
    if global_step is not None:
        logger.info(f"***** Running evaluation on {mode} dataset ({global_step} step) *****")
        pass
    else:
        logger.info(f"***** Running evaluation on {mode} dataset *****")
        pass

    device = config.device
    no_cuda = config.no_cuda

    eval_loss = 0.0
    eval_step = 0
    preds = None
    out_label_ids = None

    #for batch in progress_bar(dataloader):
    for batch in dataloader:
        model.eval()

        # this where evaluation step goes
        with torch.no_grad():
            T, TN, B, BN, V, S, A1, A2 = batch

            # text mask
            H = torch.cat([T, B], 1)
            S_ = S[:,:-1]

            HM, SM = create_masks(H, S_, config.pad_token_id)

            if not no_cuda:
                T = T.to(device)
                B = B.to(device)
                HM = HM.to(device)
                S_ = S_.to(device)
                SM = SM.to(device)
                V = V.to(device)
                A1 = A1.to(device)
                A2 = A2.to(device)
                pass

            preds = model(T, TN, B, BN, HM, S_, SM, V, A1, A2)
            ys = S[:, 1:].contiguous().view(-1)
            if not no_cuda:
                ys = ys.to(device)
                pass

            loss = F.cross_entropy(preds.view(-1, preds.size(-1)), ys, ignore_index=config.pad_token_id)
            eval_loss += loss.item()
            pass

        eval_step += 1
        pass

    eval_loss = eval_loss / eval_step
    logger.info(f'eval_loss: {eval_loss:.4f}, eval_steps:{eval_step}')

    pass


def get_model_dir(config):
    fnm = whoami()

    now = datetime.datetime.now(pytz.timezone(config.timezone))
    now_str = f'{now.year:0>4}-{now.month:0>2}-{now.day:0>2}_{now.hour:0>2}:{now.minute:0>2}:{now.second:0>2}'
    return os.path.join(config.model_dir, now_str)


def train_model(config, model, train_dataset, test_dataset=None):
    fnm = whoami()

    train_dataloader = DataLoader(train_dataset, \
                                  batch_size=config.train_batch_size, \
                                  num_workers=config.n_workers_for_dataloader, \
                                  collate_fn=collate_func, \
                                  sampler=BatchSampler(config.train_batch_size, len(train_dataset)))

    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, \
                                 batch_size=config.eval_batch_size, \
                                 num_workers=config.n_workers_for_dataloader, \
                                 collate_fn=collate_func, \
                                 sampler=test_sampler )
    if 0 < config.max_steps:
        t_toal = config.max_steps
        config.num_train_epochs = config.max_steps // (len(train_dataloader) // config.gradient_accumulation_steps) + 1
        pass
    else:
        t_total = len(train_dataloader) // config.gradient_accumulation_steps * config.num_train_epochs
        pass

    n_params = len([model.named_parameters()])

    no_decay = ['bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [
        {'params'      : [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': config.weight_decay},
        {'params'      : [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0
        }
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate, eps=config.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup( optimizer
                                               , num_warmup_steps=int(t_total * config.warmup_proportion)
                                               , num_training_steps=t_total
                                               )
    optimizer_path = os.path.join(config.model_name_or_path, 'optimizer.pt')
    scheduler_path = os.path.join(config.model_name_or_path, 'scheduler.pt')

    if os.path.isfile(optimizer_path) and os.path.isfile(scheduler_path):
        # Load optimizer and scheduler states
        optimizer.load_state_dict(torch.load(optimizer_path))
        scheduler.load_state_dict(torch.load(scheduler_path))
        pass

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", config.num_train_epochs)
    logger.info("  Total train batch size = %d", config.train_batch_size)
    logger.info("  Gradient Accumulation steps = %d", config.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  Logging steps = %d", config.logging_steps)
    logger.info("  Save steps = %d", config.save_steps)

    device = config.device
    no_cuda = config.no_cuda

    model_dir = get_model_dir(config)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        pass

    global_step = 0
    tr_loss = 0.0

    model.zero_grad()

    #mb = master_bar(range(config.num_train_epochs))
    for epoch in range(config.num_train_epochs):
        #epoch_iterator = progress_bar(train_dataloader, parent=mb)
        #for step, batch in enumerate(epoch_iterator):
        for step, batch in enumerate(train_dataloader):
            model.train()

            # this where training step goes
            T, TN, B, BN, V, S, A1, A2 = batch

            # text mask
            H = torch.cat([T, B], 1)
            S_ = S[:,:-1]

            HM, SM = create_masks(H, S_, config.pad_token_id)

            if not no_cuda:
                T = T.to(device)
                B = B.to(device)
                HM = HM.to(device)
                S_ = S_.to(device)
                SM = SM.to(device)
                V = V.to(device)
                A1 = A1.to(device)
                A2 = A2.to(device)
                pass

            preds = model(T, TN, B, BN, HM, S_, SM, V, A1, A2)
            ys = S[:, 1:].contiguous().view(-1)

            if not no_cuda:
                ys = ys.to(device)
                pass

            loss = F.cross_entropy(preds.view(-1, preds.size(-1)), ys, ignore_index=config.pad_token_id)
            logger.info(f'{fnm}: epoch:{epoch:>2} / step:{step:>5} / global_step: {global_step:>5} / loss:{loss.item():.6f}')

            if 1 < config.gradient_accumulation_steps:
                loss = loss / config.gradient_accumulation_steps
                pass

            loss.backward()
            tr_loss += loss.item()

            if (step + 1) % config.gradient_accumulation_steps == 0 or \
               (len(train_dataloader) <= config.gradient_accumulation_steps \
                and (step + 1) == len(train_dataloader)):
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

                if 0 < config.logging_steps and global_step % config.logging_steps == 0:
                    if config.evaluate_test_during_training:
                        evaluate(config, model, test_dataset, 'test', global_step)
                        pass
                    pass

                if 0 < config.save_steps and global_step % config.save_steps == 0:
                    # Save model checkpoint
                    model_path = os.path.join(model_dir, f'model-{global_step:0>5}.pth')

                    torch.save(model.state_dict(), model_path)
                    logger.info(f'model saved to {model_path}')

                    torch.save(config, os.path.join(model_dir, 'config.bin'))
                    logger.info(f'config saved model to {model_path}')

                    if config.save_optimizer:
                        torch.save(optimizer.state_dict(), os.path.join(model_dir, 'optimizer.pt'))
                        torch.save(scheduler.state_dict(), os.path.join(model_dir, 'scheduler.pt'))

                        logger.info(f'optimizer and scheduler states to {model_dir}')
                        pass
                    pass

                pass

            if 0 < config.max_steps and config.max_steps < global_step:
                logger.info(f'{fnm}: config.max_steps:{config.max_steps} / global_stpes:{global_step}, break...')
                break

            pass # end of for step, batch in ...
        pass # end of for epoch in ...

    return global_step, tr_loss
            

def get_dataset(config, mode):
    data_file_path = os.path.join(config.data_dir, f'ids.{mode}.pkl')
    with open(data_file_path, 'rb') as f:
        data = pickle.load(f)
        pass

    dataset = GNN_Dataset(config, mode, data)
    return dataset


def main():
    fnm = whoami()

    init_logger()
    args = get_args()

    config_file_path = os.path.join(args.config_dir, args.config_file)
    config = load_config(config_file_path)

    set_seed(config)

    # set device (GPU / CPU)
    if config.no_cuda:
        config.device = 'cpu'
        pass
    else:
        config.device = get_gpu(print_gpu_info=True)
        pass

    # --------------
    # init. tokenizer
    # --------------
    nlp_model_type = config.model_type
    nlp_model_path = config.model_name_or_path
    do_lower_case  = config.do_lower_case

    tokenizer = ElectraTokenizer.from_pretrained(config.model_name_or_path, do_lower_case=config.do_lower_case)
    lm = ElectraModel.from_pretrained(config.model_name_or_path)  # language model for KoELECTRA-Base-v3

    if not config.no_cuda:
        lm = lm.cuda()
        pass

    config.tokenizer = tokenizer
    config.lm = lm

    config.cls_token_id = tokenizer.cls_token_id
    config.sep_token_id = tokenizer.sep_token_id
    config.pad_token_id = tokenizer.pad_token_id

    # --------------

    model = get_model(config)

    train_data_file_path = os.path.join(config.data_dir, 'ids.train.pkl')
    with open(train_data_file_path, 'rb') as f:
        train_data = pickle.load(f)
        pass

    train_dataset = get_dataset(config, 'train')
    test_dataset  = get_dataset(config, 'test')

    if config.do_train:
        global_step, tr_loss = train_model(config, model, train_dataset, test_dataset)
        logger.info(f'{fnm}: global_step:{global_step}, average loss:{tr_loss}')
        pass

    pass


if __name__ == '__main__':
    main()
    pass
