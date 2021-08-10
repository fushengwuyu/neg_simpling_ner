# author: sunshine
# datetime:2021/6/2 上午10:28

import torch.nn as nn
import torch
from transformers import AutoModel
import numpy as np
import torch.nn.functional as F


class biaffine(nn.Module):
    def __init__(self, in_size, out_size, bias_x=True, bias_y=True):
        super().__init__()
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.out_size = out_size
        self.U = torch.nn.Parameter(torch.Tensor(in_size + int(bias_x), out_size, in_size + int(bias_y)))

    def forward(self, x, y):
        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), dim=-1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), dim=-1)

        """
        batch_size,seq_len,hidden=x.shape
        bilinar_mapping=torch.matmul(x,self.U)
        bilinar_mapping=bilinar_mapping.view(size=(batch_size,seq_len*self.out_size,hidden))
        y=torch.transpose(y,dim0=1,dim1=2)
        bilinar_mapping=torch.matmul(bilinar_mapping,y)
        bilinar_mapping=bilinar_mapping.view(size=(batch_size,seq_len,self.out_size,seq_len))
        bilinar_mapping=torch.transpose(bilinar_mapping,dim0=2,dim1=3)
        """
        bilinar_mapping = torch.einsum('bxi,ioj,byj->bxyo', x, self.U, y)
        return bilinar_mapping


def batch_gather(data: torch.Tensor, index: torch.Tensor):
    length = index.shape[0]
    t_index = index.cpu().numpy()
    t_data = data.cpu().data.numpy()
    result = []
    for i in range(length):
        result.append(t_data[i, t_index[i], :])

    return torch.from_numpy(np.array(result)).to(data.device)


class SpanNER(nn.Module):
    """
        ERENet : entity relation extraction
    """

    def __init__(self, args, num_labels):
        super(SpanNER, self).__init__()
        self.num_labels = num_labels
        print('mhs with bert')
        self.bert = AutoModel.from_pretrained(args.bert_model_path, output_hidden_states=True)

        classifier_hidden = args.hidden_size * 4
        self._activator = nn.Sequential(nn.Linear(classifier_hidden, classifier_hidden),
                                        nn.Tanh(),
                                        nn.Linear(classifier_hidden, num_labels))
        self._dropout = nn.Dropout(args.dropout)

    def forward(self, token_ids, segment_ids, attention_mask):
        share_encoder = self.bert(
            input_ids=token_ids,
            attention_mask=attention_mask,
            token_type_ids=segment_ids
        )[0]

        batch_size, token_num, hidden_dim = share_encoder.size()

        ext_row = share_encoder.unsqueeze(2).expand(batch_size, token_num, token_num, hidden_dim)
        ext_column = share_encoder.unsqueeze(1).expand_as(ext_row)
        table = torch.cat([ext_row, ext_column, ext_row - ext_column, ext_row * ext_column], dim=-1)
        score_t = self._activator(self._dropout(table))

        return score_t
