import torch

from model import BiLstmCrf
from preprocess import Preprocess


# 根据训练所得模型，在测试集上进行实体抽取
def predict(model_path, save_path):
    """
    在测试集上标注的函数
    :param model_path: 模型的路径
    :param save_path: 序列标注得到的标签文件存放的路径
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pre = Preprocess()
    test_corpus = pre.preprocess_corpus('test.txt', has_tag=False)
    model = BiLstmCrf(device=device, output_size=len(pre.tag2id)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    # print(model)
    tag_list = []  # 标注结果的列表
    with torch.no_grad():
        for st in test_corpus:
            len_st = len(st)
            # 获取标签
            st = torch.tensor(st, dtype=torch.float32).unsqueeze(0).to(device)
            pred = model.forward(st, None, [len_st], is_predict=True)

            st_tag = []
            for id in pred[0]:
                # 将预测为填充的结果转换为O
                if id == 0:
                    id = 9
                st_tag.append(pre.id2tag[id])
            tag_list.append(st_tag)

    # 将预测的结果保存在磁盘中
    with open(save_path, 'w', encoding='UTF-8') as f:
        for st_tag in tag_list:
            line = ' '.join(st_tag)
            f.write(line + '\n')
        f.close()


if __name__ == '__main__':
    predict('data/model/model.pth','output/test_TAG.txt')
