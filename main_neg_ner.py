# author: sunshine
# datetime:2021/6/2 上午10:28

import argparse
from src.train import Trainer
from src.data_loader import SpanDataset
from transformers import BertTokenizer
from src.utils import load_data
import json


def get_args():
    parser = argparse.ArgumentParser()

    # file parameters
    parser.add_argument("--data_dir", default="dataset/o_data", type=str, required=False)
    parser.add_argument("--output", default="output", type=str, required=False,
                        help="The output directory where the model checkpoints and predictions will be written.")

    # choice parameters

    parser.add_argument('--warm_up', type=bool, default=False)
    parser.add_argument('--debug', type=bool, default=False)

    # train parameters
    parser.add_argument('--train_mode', type=str, default="train")
    parser.add_argument("--batch_size", default=4, type=int, help="Total batch size for training.")

    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--epoch_num", default=10, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--patience_stop', type=int, default=10, help='Patience for learning early stop')
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument('--device_id', type=int, default=0, help="gpu id")
    # bert parameters
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% "
                             "of training.")
    parser.add_argument("--bert_model_path", default='/home/zmw/pre_models/pytorch/albert_chinese_tiny',
                        type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                             "bert-base-multilingual-cased, bert-base-chinese.")

    # model parameters
    parser.add_argument("--max_len", default=512, type=int)
    parser.add_argument('--hidden_size', type=int, default=312)
    parser.add_argument('--num_labels', type=int, default=2)
    parser.add_argument('--dropout', type=int, default=0.1)
    args = parser.parse_args()
    return args


def build_dataset(args, tokenizer):
    """
    数据处理
    :return:
    """
    train_path = args.data_dir + '/train_data.txt'
    valid_path = args.data_dir + '/valid_data.txt'

    train_data = load_data(train_path)
    valid_data = load_data(valid_path)

    labels = ['O', 'dis', 'sym', 'pro', 'equ', 'dru', 'ite', 'bod', 'dep', 'mic']
    label2id = {l: idx for idx, l in enumerate(labels)}

    train_loader = SpanDataset(train_data, label2id, tokenizer=tokenizer, max_len=args.max_len).get_data_loader(
        batch_size=args.batch_size, shuffle=True)
    valid_loader = SpanDataset(valid_data, label2id, tokenizer=tokenizer, max_len=args.max_len,
                               is_dev=True).get_data_loader(
        batch_size=args.batch_size, shuffle=False)

    return [train_loader, valid_loader], label2id


def main():
    # 准备参数
    args = get_args()

    tokenizer = BertTokenizer.from_pretrained(args.bert_model_path)
    # 处理数据
    data_loader, label2id = build_dataset(args, tokenizer)

    # 构建trainer
    # args, data_loader, label2id, tokenizer
    trainer = Trainer(
        args=args,
        data_loader=data_loader,
        label2id=label2id
    )

    if args.train_mode == 'train':
        trainer.train(args)

    elif args.train_mode == 'eval':
        eval_result = trainer.evaluate()
        print(eval_result)

    else:
        ...


if __name__ == '__main__':
    main()
