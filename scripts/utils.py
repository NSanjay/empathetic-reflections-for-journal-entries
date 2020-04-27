import argparse
import glob
import logging
import os
import json
import random
import re
import shutil
from typing import Dict, List, Tuple
from pathlib import *

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import (
    MODEL_WITH_LM_HEAD_MAPPING,
    WEIGHTS_NAME,
    CONFIG_NAME,
    AdamW,
    AutoConfig,
    AutoModelWithLMHead,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
)

logger = logging.getLogger(__name__)

class CustomLineByLineTextDataset(Dataset):

    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int, sep_token="<sep>"):
        logger.info("Creating features from dataset file at %s", file_path)

        with open(file_path, encoding="utf-8") as f:
            lines = []
            i = 0
            for line in f:
                line = json.loads(line)
                question_text = line.get("question_text")
                answer_text = line.get("answer_text")
                example_text = question_text + " " + sep_token + " " + answer_text
                # if i < 10:
                #     logger.info(f"{example_text}")
                lines.append(example_text)
                i += 1

        batch_encoding = tokenizer.batch_encode_plus(lines, add_special_tokens=True, max_length=block_size)
        self.examples = batch_encoding["input_ids"]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:
        return torch.tensor(self.examples[i], dtype=torch.long)

class GenerateTextDataset(Dataset):

    def __init__(self, tokenizer: PreTrainedTokenizer, example: str, block_size: int, sep_token="<sep>"):
        batch_encoding = tokenizer.batch_encode_plus([example], add_special_tokens=True, max_length=block_size)
        self.examples = batch_encoding["input_ids"]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:
        return torch.tensor(self.examples[i], dtype=torch.long)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)