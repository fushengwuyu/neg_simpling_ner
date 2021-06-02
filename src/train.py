# author: sunshine
# datetime:2021/6/2 上午10:28

import logging
import sys
import time
from warnings import simplefilter
import torch
from tqdm import tqdm
import torch.nn as nn
from src.utils import sequence_padding, flat_list
from src.model import SpanNER
from transformers import AdamW, get_linear_schedule_with_warmup, BertTokenizerFast

simplefilter(action='ignore', category=FutureWarning)
logger = logging.getLogger(__name__)


class Trainer(object):

    def __init__(self, args, data_loader, examples, spo_conf, tokenizer):

        self.args = args
        self.tokenizer = tokenizer
        self.device = torch.device("cuda:{}".format(args.device_id) if torch.cuda.is_available() else "cpu")
        self.n_gpu = torch.cuda.device_count()

        self.id2rel = {item: key for key, item in spo_conf.items()}
        self.rel2id = spo_conf

        if self.n_gpu > 0:
            torch.cuda.manual_seed_all(args.seed)
        self.model = SpanNER(args, len(spo_conf))

        self.model.to(self.device)
        if args.train_mode == "eval":
            self.resume(args)
        logging.info('total gpu num is {}'.format(self.n_gpu))

        self.train_dataloader = data_loader
        self.train_examples, self.dev_examples = examples
        self.optimizer, self.schedule = self.set_optimizer(args, self.model,
                                                           train_steps=(int(
                                                               len(
                                                                   self.train_examples) / args.train_batch_size) + 1) * args.epoch_num)
        self._criterion = nn.NLLLoss()

    def set_optimizer(self, args, model, train_steps=None):
        param_optimizer = list(model.named_parameters())
        param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
        num_train_steps = args.epoch_num * train_steps
        schedule = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=num_train_steps * args.warmup_proportion,
            num_training_steps=num_train_steps
        )
        return optimizer, schedule

    def train(self, args):

        best_loss = 1e10
        patience_stop = 0
        self.model.train()
        step_gap = 20
        for epoch in range(int(args.epoch_num)):

            global_loss, gap_loss = 0.0, 0.0

            for step, batch in tqdm(enumerate(self.train_dataloader), mininterval=5,
                                    desc=u'training at epoch : %d ' % epoch, leave=False, file=sys.stdout):

                loss = self.forward(batch)
                global_loss += loss
                if step % step_gap == 0:
                    gap_loss += loss
                    current_loss = gap_loss / step_gap
                    print(
                        u"step {} / {} of epoch {}, train/loss: {}".format(step, len(self.train_dataloader),
                                                                           epoch, current_loss))
                    gap_loss = 0.0
            epoch_loss = global_loss / len(self.train_dataloader)
            if epoch_loss < best_loss:
                best_loss = epoch_loss

                logging.info("** ** * Saving fine-tuned model ** ** * ")
                model_to_save = self.model.module if hasattr(self.model,
                                                             'module') else self.model  # Only save the model it-self
                output_model_file = args.output + "/pytorch_model.bin"
                torch.save(model_to_save.state_dict(), str(output_model_file))
                patience_stop = 0
            else:
                patience_stop += 1
            if patience_stop >= args.patience_stop:
                return

    def resume(self, args):
        resume_model_file = args.output + "/pytorch_model.bin"
        logging.info("=> loading checkpoint '{}'".format(resume_model_file))
        checkpoint = torch.load(resume_model_file, map_location='cpu')
        self.model.load_state_dict(checkpoint)

    def forward(self, batch, eval=False):

        batch = tuple(t.to(self.device) for t in batch)
        if not eval:
            token_ids, segment_ids, attention_mask, positions, label = batch
            score = self.model(token_ids, segment_ids, attention_mask)
            positions = flat_list(positions)
            flat_s = torch.cat([score[[i], j, k] for i, j, k in positions], dim=0)
            loss = self._criterion(torch.log_softmax(flat_s, dim=-1), label)

            if self.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.

            loss.backward()
            loss = loss.item()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.schedule.step()
            return loss

    def extract_entity(self, text):
        ...

    def evaluate(self):
        self.model.eval()
        tp, fp, fn = 0, 0, 0
        for d in self.dev_examples[:10]:
            text = d.context
            gold = d.gold_answer
            pred = self.extract_entity(text)
            tp_tmp = gold & pred
            tp += tp_tmp
            fp += fp_tmp
            fn += fn_tmp
        p = tp / (tp + fp) if tp + fp != 0 else 0
        r = tp / (tp + fn) if tp + fn != 0 else 0
        f = 2 * p * r / (p + r) if p + r != 0 else 0
        self.model.train()
        return {'f1': f, "recall": r, "precision": p}

    def predict(self, texts):
        tokenizer = BertTokenizerFast.from_pretrained(self.args.bert_model)
        inputs = tokenizer(texts, padding='longest', return_offsets_mapping=True)
        batch_token_ids = sequence_padding(inputs['input_ids'])
        batch_segment_ids = sequence_padding(inputs['token_type_ids'])

        batch_mapping = inputs['offset_mapping']

        # 构造eval
        batch_eval = []
        p_id = 0
        for text, mapping, token_ids in zip(texts, batch_mapping, batch_token_ids):
            tok_to_orig_start_index = [m[0] for m in mapping]
            tok_to_orig_end_index = [m[1] for m in mapping]
            batch_eval.append(
                Example(
                    p_id=p_id,
                    context=text,
                    bert_tokens=tokenizer.tokenize(text, add_special_tokens=False),
                    gold_answer=None,
                    spoes=None,
                    sub_entity_list=None,
                    tok_to_orig_start_index=tok_to_orig_start_index,
                    tok_to_orig_end_index=tok_to_orig_end_index,
                )
            )
            p_id += 1
        p_ids = torch.tensor(list(range(len(texts))), dtype=torch.long)
        answer_dict = {i: [[], [], {}] for i in range(len(batch_eval))}

        last_time = time.time()
        with torch.no_grad():
            cursor = 0
            for i in tqdm(range(len(texts)), mininterval=5, leave=False, file=sys.stdout):
                if i % 32 == 0 or i == len(texts):
                    self.forward([p_ids[cursor:i], batch_token_ids[cursor: i], batch_segment_ids[cursor: i]], 'valid',
                                 eval=True,
                                 eval_file=batch_eval,
                                 answer_dict=answer_dict)
                    cursor = i
        used_time = time.time() - last_time
        # logging.info('chosen {} took : {} sec'.format(chosen, used_time))

        self.convert2ressult(batch_eval, answer_dict)

        spos_predicts = []
        for key in answer_dict.keys():
            triple_pred = answer_dict[key][1]
            spos_predicts.append(triple_pred)
        return spos_predicts
