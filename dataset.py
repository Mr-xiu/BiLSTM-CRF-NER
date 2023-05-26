import torch
from torch.nn.utils.rnn import pad_sequence

from torch.utils.data import Dataset
class NerDataset(Dataset):
    def __init__(self, corpus_list, tags_list):
        """
        初始化方法
        :param corpus_list: 转换为词向量形式的语料
        :param tags_list: 转换为id形式的标签
        """
        self.corpus_list = corpus_list
        self.tags_list = tags_list

    def __len__(self):
        return len(self.tags_list)

    def __getitem__(self, item):
        """
        :param item: 获取句子的下标
        :return: 返回转换为tensor后的句子，标签以及句子的长度
        """
        corpus = torch.tensor(self.corpus_list[item], dtype=torch.float32)
        tags = torch.tensor(self.tags_list[item], dtype=torch.int32)
        return corpus, tags, len(self.tags_list[item])


def my_collate_fn(batch):
    corpus_list = []
    tag_list = []
    len_list = []
    for bt in batch:
        corpus_list.append(bt[0])
        tag_list.append(bt[1])
        len_list.append(bt[2])
    corpus_list = pad_sequence(corpus_list, batch_first=True, padding_value=0.0)
    tag_list = pad_sequence(tag_list, batch_first=True, padding_value=0)
    return corpus_list, tag_list, len_list
