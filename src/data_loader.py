# author: sunshine
# datetime:2021/6/2 上午10:28


from functools import partial

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from src.utils import sequence_padding, fine_grad_tokenize, flat_list


class SpanDataset(Dataset):
    def __init__(self, data, label2id, tokenizer=None, max_len=128, neg_rate=0.7):
        super(SpanDataset).__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label2id = label2id
        self.neg_rate = neg_rate

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def _create_collate_fn(self):
        def collate(examples):
            """
                {
                    'text': '（5）房室结消融和起搏器植入作为反复发作或难治性心房内折返性心动过速的替代疗法。',
                    'label': [['3', '7', 'pro'], ['9', '13', 'pro'], ['16', '33', 'dis']]
                }

            """

            batch_token_ids, batch_segment_ids, batch_attention_mask, batch_position, batch_label = [], [], [], [], []

            for idx, d in enumerate(examples):
                text = d['text']
                tokens = fine_grad_tokenize(text, self.tokenizer)
                inputs = self.tokenizer.encode_plus(text=tokens)

                token_ids = inputs['input_ids']
                segment_ids = inputs['token_type_ids']
                attention_mask = inputs['attention_mask']

                positions = [(l[0], l[1]) for l in d['label']]
                labels = [self.label2id[l] for l in d['label']]

                neg_positions = self.generate_whole_label(positions=positions, length=len(text))

                batch_token_ids.append(token_ids)
                batch_segment_ids.append(segment_ids)
                batch_attention_mask.append(attention_mask)
                batch_position.append(positions + neg_positions)
                batch_label.append(labels + [0] * len(neg_positions))

            # padding
            batch_token_ids = sequence_padding(batch_token_ids)
            batch_segment_ids = sequence_padding(batch_segment_ids)
            batch_attention_mask = sequence_padding(batch_attention_mask)
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
