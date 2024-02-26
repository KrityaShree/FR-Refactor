from . import seq2seq_funcs

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn import CrossEntropyLoss, MSELoss

from tqdm import tqdm, trange
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score
from random import random, randrange, randint, shuffle, choice, sample

#prep for phase 3
from pytorch_pretrained_bert.modeling import BertForMaskedLM, BertConfig 
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam


import sys
import os

sys.path.append(os.path.join("/seq2seq_funcs.py")) #saves from calling funcs in diff files multple times

#equivalent to main function for seq2seq_funcs. 
#changed parameters to somewhat match calls by main program

def train():
	if args.local_rank == -1 or args.no_cuda
	device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))


    # do not remove grad_accumulation. weird error triggers unexpectedly midway training
    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    #bs = 128 (tune for best results)
    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps


    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    tokenizer = BertTokenizer(vocab_file=args.vocab_file)

    train_examples = None
    numOptSteps = None
    vocab_list = []
    with open(args.vocab_file, 'r') as fr:
         for line in fr:
             vocab_list.append(line.strip("\n"))

    if args.do_train:
        train_examples = create_examples(data_path=args.pretrain_train_path,
                                         max_seq_len=args.max_seq_len,
                                         masked_lm_prob=args.masked_lm_prob,
                                         max_preds=args.max_preds,
                                         vocab_list=vocab_list)
        numOptSteps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            numOptSteps = numOptSteps // torch.distributed.get_world_size()

    model = BertForMaskedLM(config=BertConfig.from_json_file(args.bert_config_json))
    if args.fp16: #mixed precision training 
        model.half()
    model.to(device)
    
###########################################
#code taken and edited from github/abhiyal/deconstructed-seq2seq

    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optGrpdParams,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)

    else:
        optimizer = BertAdam(optGrpdParams,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=numOptSteps)

##############################################

    paramOpt = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optGrpdParams = [
        {'params': [p for n, p in paramOpt if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in paramOpt if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    global_step = 0
    best_loss = 100000

    if args.do_train:
        train_features = convert_examples_to_features(
            train_examples, args.max_seq_len, tokenizer)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)

        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        model.train()
        for e in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                loss = model(input_ids, segment_ids, input_mask, label_ids)

                if n_gpu > 1:
                    loss = loss.mean() 
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        #modify learning rate with special warm up BERT uses
                        #if args.fp16 is False, BertAdam is used which handles this automatically
                        lr_this_step = args.learning_rate * warmup_linear(global_step / numOptSteps,
                                                                          args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
                if nb_tr_steps > 0 and nb_tr_steps % 100 == 0:
                    logger.info("===================== -epoch %d -train_step %d -train_loss %.4f\n" % (e, nb_tr_steps, tr_loss / nb_tr_steps))

                model.eval()
                eval_loss = 0
                nb_eval_steps = 0
                for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
                    input_ids = input_ids.to(device)
                    input_mask = input_mask.to(device)
                    segment_ids = segment_ids.to(device)
                    label_ids = label_ids.to(device)

                    with torch.no_grad():
                        loss = model(input_ids, segment_ids, input_mask, label_ids)

                    eval_loss += loss.item()
                    nb_eval_steps += 1
                    
                if nb_tr_steps > 0 and nb_tr_steps % 2000 == 0:
                eval_examples = create_examples(data_path=args.pretrain_dev_path,
                                         max_seq_len=args.max_seq_len,
                                         masked_lm_prob=args.masked_lm_prob,
                                         max_preds=args.max_preds,
                                         vocab_list=vocab_list)
                eval_features = convert_examples_to_features(
                    eval_examples, args.max_seq_len, tokenizer)
                all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
                all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
                all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
                all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
                eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

                # run preds for feauture docs
                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)


                eval_loss = eval_loss / nb_eval_steps
                    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                    output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
                    torch.save(model_to_save.state_dict(), output_model_file)
                    best_loss = eval_loss
                logger.info("============================ -epoch %d -train_loss %.4f -eval_loss %.4f\n"% (e, tr_loss / nb_tr_steps, eval_loss))

