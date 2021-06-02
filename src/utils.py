# author: sunshine
# datetime:2021/6/2 上午10:33

import torch
import numpy as np


def load_data(path):
    """
    data analysis
    :param path:
    :return:
    （5）房室结消融和起搏器植入作为反复发作或难治性心房内折返性心动过速的替代疗法。|||3    7    pro|||9    13    pro|||16    33    dis|||

    """
    D = []
    i = 0
    with open(path, 'r', encoding='utf-8') as rd:
        for l in rd:
            fields = l.strip('\n').split('|||')
            if len(fields[0]) > 510:
                continue
            D.append({"text": fields[0], "label": [i.split('    ') for i in fields[1:] if i]})
    print(i)
    return D


def sequence_padding(inputs, length=None, padding=0, is_float=False, is_int=False):
    """Numpy函数，将序列padding到同一长度

    """
    if length is None:
        length = max([len(x) for x in inputs])

    outputs = np.array([
        np.concatenate([x, [padding] * (length - len(x))])
        if len(x) < length else x[:length] for x in inputs
    ])
    if is_float:
        out_tensor = torch.tensor(outputs, dtype=torch.float32)
    elif is_int:
        out_tensor = torch.tensor(outputs, dtype=torch.int32)
    else:
        out_tensor = torch.tensor(outputs, dtype=torch.long)
    return torch.tensor(out_tensor)


def search(patten, sequence):
    n = len(patten)
    for i in range(len(sequence)):
        if sequence[i: i + n] == patten:
            return i
    return -1


def fine_grad_tokenize(text, tokenizer):
    """
    char-level tokenize
    :param text:
    :param tokenizer:
    :return:
    """

    tokens = []
    for _ch in text:
        if _ch in [' ', '\t', '\n']:
            tokens.append(['[BLANK]'])
        else:
            if not len(tokenizer.tokenize(_ch)):
                tokens.append('[INV]')
            else:
                tokens.append(_ch)


def flat_list(nest_list):
    """
    :param nest_list:
    :return:
    """
    return [item for sublist in nest_list for item in sublist]


if __name__ == '__main__':
    train_data = load_data('../dataset/o_data/train_data.txt')
    dev_data = load_data('../dataset/o_data/val_data.txt')
    print(train_data[0])
    l = [len(t['text']) for t in train_data]
    d = [len(t['text']) for t in dev_data]
    from collections import Counter

    print(Counter(d + l))
    print(len(train_data), len(dev_data))
