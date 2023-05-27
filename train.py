import os
import matplotlib.pyplot as plt
import torch
from preprocess import Preprocess
from model import BiLstmCrf
from dataset import my_collate_fn, NerDataset
from tqdm import tqdm
from torch.utils.data import DataLoader


def get_f1(pred, label):
    """
    计算f1值的方法
    :param pred: 预测的tag
    :param label: 真实的tag
    :return: 计算得到的f1值
    """
    assert len(pred) == len(label)
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(len(pred)):
        # 若真实值不为O
        if label[i] != 9:
            if pred[i] == label[i]:
                TP += 1
            else:
                FN += 1
        elif pred[i] != 9:
            FP += 1
        else:
            TN += 1
    # 准确率
    precision = 0 if (TP + FP) == 0 else TP / (TP + FP)
    # 召回率
    recall = 0 if (TP + FN) == 0 else TP / (TP + FN)
    f1 = 0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    return f1


def train(max_epoch, batch_size, save_path='data/model/model.pth'):
    """
    训练的函数
    :param max_epoch: 最大训练轮数
    :param batch_size: 每个batch的长度
    :param save_path: 保存模型的路径
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pre = Preprocess()
    train_corpus, train_tags = pre.preprocess_corpus('train.txt', 'train_TAG.txt', has_tag=True)
    dev_corpus, dev_tags = pre.preprocess_corpus('dev.txt', 'dev_TAG.txt', has_tag=True)
    train_dataset = NerDataset(train_corpus, train_tags)
    dev_dataset = NerDataset(dev_corpus, dev_tags)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=my_collate_fn, shuffle=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, collate_fn=my_collate_fn, shuffle=True)
    # 模型
    model = BiLstmCrf(device=device, output_size=len(pre.tag2id)).to(device)
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    f1_list = []
    loss_list = []

    for epoch in range(max_epoch):
        print(f'第{epoch + 1}轮...')
        print('开始训练...')
        # 训练
        tqdm_obj = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        for i, (inputs, labels, lens) in tqdm_obj:
            loss = model.forward(inputs.to(device), labels.to(device), lens, is_test=False)
            optimizer.zero_grad()
            loss.backward(torch.ones_like(loss))
            optimizer.step()
            loss = torch.mean(loss).item()
            tqdm_obj.set_postfix(loss=loss)
            # 每256个batch记录一次loss信息
            if (i + 1) % 1000 == 0:
                loss_list.append(loss)


        # 验证
        print('开始验证...')
        with torch.no_grad():
            pred_list = []
            label_list = []
            for i, (inputs, labels, lens) in enumerate(dev_dataloader):
                # 预测的pred为list格式
                pred = model.forward(inputs.to(device), labels.to(device), lens, is_test=True)
                # 需要将labels也转换为list格式
                labels = labels.tolist()
                for j in range(len(pred)):
                    pred_list.extend(pred[j][:lens[j]])
                    label_list.extend(labels[j][:lens[j]])
            f1 = get_f1(pred_list, label_list)
            print(f'第{epoch + 1}轮训练结束，在发展集上非O标签的F1值为{f1}')
            f1_list.append(f1)

            if epoch > 1:
                # 提前终止
                if f1_list[-2] > f1 and f1_list[-2] > 0.9:
                    print('提前终止！！！')
                    break
            # 更新模型
            if epoch > 0:
                os.remove(save_path)
            torch.save(model.state_dict(), save_path)

    # 绘制loss图与f1图
    plt.plot(loss_list)
    plt.title('loss')
    plt.savefig('data/imgs/loss.jpg')
    plt.close()

    plt.plot(f1_list)
    plt.title('F1')
    plt.savefig('data/imgs/F1.jpg')
    plt.close()


if __name__ == '__main__':
    train(max_epoch=20, batch_size=512)
