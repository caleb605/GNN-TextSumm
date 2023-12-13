# -*- coding: utf-8 -*-

import os, sys
import glob
import json

import logging

from pykospacing import Spacing
import kss

from misc import whoami


MAX_TEXT_TOKENS = 1000
MAX_SUMM_TOKENS = 200
MAX_SENT_TOKENS = 80 # refert to stats. below
MAX_SENTS = 40       # refert to stats. below

N_PADDED_ALL = MAX_SENT_TOKENS + MAX_SENTS

#
# distribution of length of sentences in 320,182 training text
#
#   number of sentences below length of 32 : 313,855 : 98.02%
#   number of sentences below length of 40 : 318,660 : 99.52%
#   number of sentences below length of 48 : 319,828 : 99.89%
#   number of sentences below length of 56 : 320,067 : 99.96%
#   number of sentences below length of 64 : 320,104 : 99.98%
#
# mean         : 15.23
# std          : 18.58
# min          :  3
# 1st quantile : 10
# 2nd quantile : 14
# 3rd quantile : 19
# max          : 3089
#
# --------------------------------------------------------
# n_tokens in a sent upto 60: 4068213: 84.30853318086197 %
# n_tokens in a sent upto 64: 4392301: 91.02484422492952 %
# n_tokens in a sent upto 68: 4392303: 91.02488567238234 %
# n_tokens in a sent upto 72: 4648322: 96.33055338359390 %
# n_tokens in a sent upto 76: 4654613: 96.46092634642568 %
# n_tokens in a sent upto 80: 4825376: 99.99977203900951 %
# n_tokens in a sent upto 88: 4825385: 99.99995855254718 %
# --------------------------------------------------------
#


spacer = Spacing()

logger = logging.getLogger(__name__)


def get_data_from_ai_hub(config, mode='train', text_list=None, summ_list=None):
    fnm = whoami()

    text_dir = config.text_dir

    data_path_dict = {
            'train': [f'{text_dir}/1.Training/법률문서/train.jsonl'
                     ,f'{text_dir}/1.Training/사설잡지/train.jsonl'
                     ,f'{text_dir}/1.Training/신문기사/train.jsonl'
                     ]
           ,'test' : [f'{text_dir}/2.Validation/법률문서/dev.jsonl'
                     ,f'{text_dir}/2.Validation/사설잡지/dev.jsonl'
                     ,f'{text_dir}/2.Validation/신문기사/dev.jsonl'
                     ]
    }
    file_paths = data_path_dict.get(mode, None)

    if file_paths is None:
        warning = f'mode({fnm}) not valid. "train" or "test" must be specified as mode'
        raise ValueError(warnign)

    if text_list is None:
        text_list = list()
        pass

    if summ_list is None:
        summ_list = list()
        pass

    for file_path in file_paths:
        with open(file_path, 'r') as f:
            for i, line in enumerate(f.readlines()):
                article = json.loads(line)
                text = article.get('article_original', None)
                summ = article.get('abstractive', None)

                if not text or not summ:
                    logger.info(f'{i + 1}th line: not valid in {file_path}')
                    continue

                text_list.append(text)
                summ_list.append(summ)
                pass
            pass
        pass
    return text_list, summ_list


def get_data_from_sci_news_50(text_list=None, summ_list=None):
    data_dir = '/DATA/0106.TextSummarization/sci-news-sum-kr-50/data'

    if text_list is None:
        text_list = list()
        pass

    if summ_list is None:
        summ_list = list()
        pass

    for file_path in glob.glob(os.path.join(data_dir, '*.json')):
        with open(file_path, 'r') as f:
            for i, line in enumerate(f.readlines()):
                article = json.loads(line)

                sentences = article.get('sentences', None)
                for i in range(len(sentences)):
                    sent = sentences[i]
                    sent = sent.strip()
                    if sent[-1] != '.':
                        sent += '.'
                        pass
                    sentences[i] = sent
                    pass

                text = ' '.join(sentences)
                summ = article.get('title', None)

                if not text or not summ:
                    logger.info(f'{i + 1}th line: not valid in {file_path}')
                    continue

                text_list.append(text)
                summ_list.append(summ)
                pass
            pass
        pass
    return text_list, summ_list


def preprocess_text(tokenizer, text, split=False):
    if isinstance(text, list):
        text = ' '.join(text)
        pass

    text = spacer(text)

    if split:
        sent_list = kss.split_sentences(text)

        token_lists = list()
        n_tokens = 0
        for sent in sent_list:
            sent_regularized = '[CLS] ' + sent + ' [SEP]'
            tokens = tokenizer.tokenize(sent_regularized)
            n_tokens += len(tokens)
            token_lists.append(token_lists)
            pass
        pass
    else:
        sent_regularized = '[CLS] ' + text + ' [SEP]'
        tokens = tokenizer.tokenize(sent_regularized)
        token_lists = [tokens]
        n_tokens = len(tokens)
        pass

    return token_lists, n_tokens


def get_tokens(tokenizer, text, ids=True):
    if isinstance(text, list):
        text = ' '.join(text)
        pass

    text = spacer(text)

    sentences = ' [SEP] '.join(kss.split_sentences(text))
    tokens = tokenizer.tokenize(sentences)

    if ids:
        ids = tokenizer.convert_tokens_to_ids(tokens)
        pass
    else:
        ids = None

    return tokens, ids



def main():
    pass


if __name__ == '__main__':
    main()
    pass
