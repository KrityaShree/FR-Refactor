#bin/bash/python
 CUDA_VISIBLE_DEVICES=[0,1]

from __future__ import absolute_import, division, print_function

#import pretraining_args as args
import csv
import logging
import os
import random
random.seed(args.seed)
import sys
from glob import glob

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn import CrossEntropyLoss
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score

from tqdm import tqdm, range
from random import random, randrange, randint, shuffle, choice, sample

#from config.yaml
WEIGHTS_NAME = "file_utils/weights.pt"
CONFIG_NAME = "file_utils/config.yaml"

logger = logging.getLogger(__name__)

def warmup_linear(x, warmup=2e-3):
	if x<warmup:
		return x/warmup
	return 1.0-x

class InputFeatures(object):
	def __init__(self, input_ids, input_mask, segment_ids, label_id):
		self.input_mask = input_mask
		self.input_ids = input_ids
		self.segment_ids = segment_ids
		self.label_id = label_id

	def create_maskedLM_preds(tokens, maskedLM_prob, max_preds, vocab_list):
		indices = []
		for (i,token) in enumerate(tokens):
			if token == "[CLS]" or token == "[SEP]":
				continue
			indices.append(i)

	shuffle(indices)
	maskedIndices = sorted(sample(indices, numMask))
	maskedTokenLabels = []
	for index in maskedIndices:
		if random()<0.75:
			maskedToken = "[MASK]"
		elif random()<0.5:
				maskedToken = tokens[index]
		else:
			maskedToken = choice(vocab_list)
		maskedTokenLabels.append(index)
		tokens[index] = maskedToken


	def create_examples(data_path, max_seq_len, maskedLM_prob, max_preds, vocab_list):
		examples = []
		    max_num_tokens = max_seq_length - 2
    	fr = open(data_path, "r")
    	for (i, line) in tqdm(enumerate(fr), desc="creating example"):
	        tokens_a = line.strip("\n").split()[:max_num_tokens]
	        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
	        segment_ids = [0 for _ in range(len(tokens_a) + 2)]
	        if len(tokens_a) < 5:
	            continue
	        tokens, masked_lm_positions, masked_lm_labels = create_maskedLM_preds(
	            tokens, maskedLM_prob, max_preds, vocab_list)
	        example = {
	            "tokens": tokens,
	            "segment_ids": segment_ids,
	            "masked_lm_positions": masked_lm_positions,
	            "masked_lm_labels": masked_lm_labels}
	        examples.append(example)
	    fr.close()
	    return examples

	def convert_examples_to_features(examples, max_seq_len, tokenizer):
		features = []
		for i, example in tqdm(enumerate(examples), desc="converting feature"):
			tokens = example["tokens"]
			segment_ids = example["segment_ids"]
			masked_lm_positions = example["masked_lm_labels"]
			assert(len(tokens)) == len(segment_ids) <= max_seq_len #impromptu test

			#define intrinsic values -> shift in phase 3

			input_ids = tokenizer.convert_tokens_to_ids(tokens)
			masked_label_ids = tokenizer.convert_tokens_to_ids(masked_lm_labels, masked_lm_positions)

			inputArray = np.zeros(max_seq_len, dtype=np.int)
			inputArray = [:len(input_ids)] = 1

			maskArray = np.zeros(max_seq_len, dtype=np.bool)
			maskArray = [:len(input_ids)] =1

			segmentArray = np.zeros(max_seq_len, dtype=np.bool)
			segmentArray = [:len(segment_ids)] = segment_ids

			LM_labelArray = np.full(max_seq_len, dtype=np.int, fill_value=-1)
			LM_labelArray[masked_lm_positions] = masked_label_ids

			#explicitly define features for seqmodeling -> port to BERT terminology in phase 3

			feature = InputFeatures(input_ids=inputArray, input_mask=maskArray, segment_ids=segmentArray, label_id=LM_labelArray)
			features.append(feature)

			#for iter(logger.getStep()) in range 25:
				#logger.info("input_ids: %s\ninput_mask:%s\nsegment_ids:%s\nlabel_id:%s" %(inputArray, maskArray, segmentArray, LM_labelArray))

				