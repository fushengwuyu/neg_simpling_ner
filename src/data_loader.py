# author: sunshine
# datetime:2021/6/2 上午10:28


from functools import partial

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from src.utils import sequence_padding, fine_grad_tokenize, flat_list


class SpanDataset(Dataset):
    def __init__(self, data, label2id, tokenizer=None, max_len=128, neg_rate=0.7, is_dev=False):
        super(SpanDataset).__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label2id = label2id
        self.neg_rate = neg_rate
        self.is_dev = is_dev

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def _create_collate_fn(self):
        def find_index(offset_mapping, index, is_start=True):
            for i, offset in enumerate(offset_mapping):
                if is_start:
                    if offset[0] <= index < offset[1]:
                        return i
            return -1

        def collate(examples):

            batch_token_ids, batch_segment_ids, batch_attention_mask, batch_position, batch_label = [], [], [], [], []
            lengths = []
            for idx, d in enumerate(examples):
                text, entities = d
                inputs = self.tokenizer(text, return_offsets_mapping=True)

                token_ids = inputs['input_ids']
                segment_ids = inputs['token_type_ids']
                attention_mask = inputs['attention_mask']
                offset_mapping = inputs['offset_mapping']
                lengths.append(len(token_ids))

                pos_positions = [(find_index(offset_mapping, l[0]), find_index(offset_mapping, l[1] - 1)) for l in
                                 entities]
                labels = [self.label2id[l[2]] for l in entities]

                neg_positions = self.generate_whole_label(positions=pos_positions, length=len(token_ids))

                batch_token_ids.append(token_ids)
                batch_segment_ids.append(segment_ids)
                batch_attention_mask.append(attention_mask)

                batch_position.extend([(idx,) + p for p in pos_positions + neg_positions])
                if not self.is_dev:
                    batch_label.extend(labels + [0] * len(neg_positions))
                else:
                    batch_label.extend([[idx, p[0], p[1], l] for p, l in zip(pos_positions, labels)])

            # padding
            batch_token_ids = sequence_padding(batch_token_ids)
            batch_segment_ids = sequence_padding(batch_segment_ids)
            batch_attention_mask = sequence_padding(batch_attention_mask)

            if self.is_dev:
                return [batch_token_ids, batch_segment_ids, batch_attention_mask, lengths, batch_label]
            else:
                batch_label = torch.tensor(batch_label, dtype=torch.long)
                return [batch_token_ids, batch_segment_ids, batch_attention_mask, batch_position, batch_label]

        return partial(collate)

    def generate_whole_label(self, positions, length):

        neg_positions = []
        neg_num = int(length * self.neg_rate) + 1

        candies = flat_list([[(i, j) for j in range(i, length) if (i, j) not in positions] for i in range(length)])

        if len(candies) > 0:
            sample_num = min(neg_num, len(candies))
            assert sample_num > 0

            np.random.shuffle(candies)
            for i, j in candies[:sample_num]:
                neg_positions.append((i, j))

        return neg_positions

    def get_data_loader(self, batch_size, num_workers=0, shuffle=False, pin_memory=False,
                        drop_last=False):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, collate_fn=self._create_collate_fn(),
                          num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last)
