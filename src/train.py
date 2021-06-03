# author: sunshine
# datetime:2021/6/2 上午10:28

import logging
from warnings import simplefilter
import torch
from tqdm import tqdm
import torch.nn as nn
from src.utils import flat_list
from src.model import SpanNER
from transformers import AdamW, get_linear_schedule_with_warmup
import numpy as np

simplefilter(action='ignore', category=FutureWarning)
logger = logging.getLogger(__name__)


class Trainer(object):

    def __init__(self, args, data_loader, label2id):

        self.args = args
        self.device = torch.device("cuda:{}".format(args.device_id) if torch.cuda.is_available() else "cpu")

        self.id2label = {item: key for key, item in label2id.items()}

        self.model = SpanNER(args, len(label2id))

        self.model.to(self.device)
        if args.train_mode != "train":
            self.resume(args)

        self.train_data_loader, self.dev_data_loader = data_loader
        self.optimizer, self.schedule = self.set_optimizer(args, self.model,
                                                           train_steps=(len(self.train_data_loader) * args.epoch_num))
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
        schedule = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=train_steps * args.warmup_proportion,
            num_training_steps=train_steps
        )
        return optimizer, schedule

    def train(self, args):

        best_f1 = 0
        self.model.train()
        step_gap, eval_gap = 20, 500
        for epoch in range(int(args.epoch_num)):

            global_loss, gap_loss = 0.0, 0.0
            t = tqdm(enumerate(self.train_data_loader))
            for step, batch in t:

                loss = self.forward(batch)
                gap_loss += loss
                if step % step_gap == 0:
                    current_loss = gap_loss / step_gap
                    msg = u"step {} / {} of epoch {}, train/loss: {}".format(step, len(self.train_data_loader),
                                                                             epoch, current_loss)
                    t.set_description(msg)
                    gap_loss = 0.0

                if (step + 1) % eval_gap == 0:
                    eval_result = self.evaluate()
                    print(eval_result)

                    if eval_result['f1'] > best_f1:
                        best_f1 = eval_result['f1']

                        self.save(args)

            eval_result = self.evaluate()
            print(eval_result)

            if eval_result['f1'] > best_f1:
                best_f1 = eval_result['f1']
                self.save(args)

    def save(self, args):
        logging.info("** ** * Saving fine-tuned model ** ** * ")
        model_to_save = self.model.module if hasattr(self.model,
                                                     'module') else self.model  # Only save the model it-self
        output_model_file = args.output + "/pytorch_model.bin"
        torch.save(model_to_save.state_dict(), str(output_model_file))

    def resume(self, args):
        resume_model_file = args.output + "/pytorch_model.bin"
        logging.info("=> loading checkpoint '{}'".format(resume_model_file))
        checkpoint = torch.load(resume_model_file, map_location='cpu')
        self.model.load_state_dict(checkpoint)

    def forward(self, batch, eval=False):

        batch = tuple(t.to(self.device) if not isinstance(t, list) else t for t in batch)
        if not eval:
            token_ids, segment_ids, attention_mask, positions, label = batch
            score = self.model(token_ids, segment_ids, attention_mask)
            # positions = flat_list(positions)
            flat_s = torch.cat([score[[i], j, k] for i, j, k in positions], dim=0)
            loss = self._criterion(torch.log_softmax(flat_s, dim=-1), label)
            loss.backward()
            loss = loss.item()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.schedule.step()
            return loss
        else:
            token_ids, segment_ids, attention_mask, lengths = batch
            log_items = self.model(token_ids, segment_ids, attention_mask)

            score_t = torch.log_softmax(log_items, dim=-1)
            val_table, idx_table = torch.max(score_t, dim=-1)

            idx_table = idx_table.cpu().numpy()
            bs, ss, es = np.where(idx_table != 0)

            entities = [[] for i in range(len(lengths))]
            for b, s, e in zip(bs, ss, es):
                l = lengths[b]
                if s > e or s >= l or e >= l:
                    continue
                entities[b].append([s, e, self.id2label[l]])

            return entities

    def evaluate(self):
        self.model.eval()
        A, B, C = 0, 0, 0
        for batch in tqdm(self.dev_data_loader):
            pred = self.forward(batch=batch[:-1], eval=True)
            label = batch[-1]

            pred = set([tuple(p) for p in pred])
            label = set([tuple(l) for l in label])

            A += len(pred & label)
            B += len(pred)
            C += len(label)

        p, r, f1 = A / B, A / C, 2 * A / (B + C)
        self.model.train()
        return {'f1': f1, "recall": r, "precision": p}
