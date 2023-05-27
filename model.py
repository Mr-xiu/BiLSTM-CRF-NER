import torch.nn as nn
import torch
from TorchCRF import CRF


class BiLstmCrf(nn.Module):
    def __init__(self,device, output_size, emb_size=100, hidden_dim=256):
        """
        模型的初始化方法
        :param device: 当前使用的设备（'gpu'或'cpu'）
        :param output_size: 输出的维度
        :param emb_size: 编码的维度
        :param hidden_dim: 隐含层的维度
        """
        super(BiLstmCrf, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_size = output_size
        self.device = device
        self.lstm = nn.LSTM(input_size=emb_size, hidden_size=self.hidden_dim // 2, num_layers=2, batch_first=True,
                            bidirectional=True)
        self.fc = nn.Linear(self.hidden_dim, self.output_size)
        self.crf = CRF(self.output_size)

    def forward(self, x, target, sentence_len, is_test=False):
        """
        前向传播
        x: 输入的向量 [batch_size, sentence_len, embedding_size]
        target: 标签向量[batch_size, sentence_len]
        sentence_len: batch 中每个句子的长度 [batch_size]
        is_test: 当前是否为训练，若为训练，返回的为loss，否则为解码的结果
        """
        mask = torch.tensor([[True if j < length else 0 for j in range(x.shape[1])]
                             for length in sentence_len], dtype=torch.bool).to(self.device)
        x, _ = self.lstm(x)
        x = self.fc(x)

        if not is_test:
            loss = -self.crf.forward(x, target, mask=mask)
            return loss
        else:
            decode = self.crf.viterbi_decode(x, mask=mask)
            return decode
