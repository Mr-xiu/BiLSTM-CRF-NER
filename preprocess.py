from gensim.models import KeyedVectors


# 预处理类
class Preprocess:
    def __init__(self, input_path='input/'):
        self.data_path = input_path
        # 标签与id相互转换的字典
        self.tag2id = {
            'PAD': 0,  # pad为填充的标签
            'B_PER': 1,
            'I_PER': 2,
            'B_T': 3,
            'I_T': 4,
            'B_ORG': 5,
            'I_ORG': 6,
            'B_LOC': 7,
            'I_LOC': 8,
            'O': 9,
        }
        self.id2tag = {}
        for tag, id in self.tag2id.items():
            self.id2tag[id] = tag
        # 全部加载到内存中，时间略长
        self.word2vector = KeyedVectors.load_word2vec_format("data/tencent_word_vector-zh-d100.w2v", binary=True)

    def read_corpus(self, corpus_name: str):
        """
        读取语料的方法
        :param corpus_name:
        :return:
        """
        result_list = []
        with open(self.data_path + corpus_name, 'r', encoding='UTF-8') as f:
            corpus_lines = f.readlines()
            for line in corpus_lines:
                result_list.append(line.split())
        return result_list

    @staticmethod
    def cut_corpus(corpus_list: list, tag_list: list = None, has_tag=False):
        new_corpus_list = []
        new_tag_list = []
        # 遍历语料的每一行
        for i in range(len(corpus_list)):
            # 每行截断的起始位置
            start_index = 0
            # 遍历每一个字符
            for j in range(len(corpus_list[i])):
                word = corpus_list[i][j]  # 当前字符
                # 如果一句话结束或者该行语料结束，则可以截断
                if word == ';' or word == '。' or j + 1 == len(corpus_list[i]):
                    new_corpus_list.append(corpus_list[i][start_index:j + 1])
                    if has_tag:
                        new_tag_list.append(tag_list[i][start_index:j + 1])
                    start_index = j + 1  # 更新下一个句子的起始index
        if has_tag:
            return new_corpus_list, new_tag_list
        else:
            return new_corpus_list

    def word_to_id(self, corpus_list):
        id_list = []
        for sentence in corpus_list:
            id_sentence = []
            for word in sentence:
                if self.word2vector.has_index_for(word):
                    id_sentence.append(self.word2vector[word])
                else:
                    id_sentence.append(self.word2vector[0])
            id_list.append(id_sentence)
        return id_list

    def tag_to_id(self, tag_list):
        id_list = []
        for sentence in tag_list:
            id_sentence = []
            for tag in sentence:
                id_sentence.append(self.tag2id[tag])
            id_list.append(id_sentence)
        return id_list

    def preprocess_corpus(self, corpus_name, tag_name='', has_tag=False):
        corpus = self.read_corpus(corpus_name)
        tags = None
        if has_tag:
            tags = self.read_corpus(tag_name)
            corpus, tags = self.cut_corpus(corpus, tags, has_tag)

        else:
            corpus = self.cut_corpus(corpus)
        corpus = self.word_to_id(corpus)
        if has_tag:
            tags = self.tag_to_id(tags)
            return corpus, tags
        return corpus


if __name__ == '__main__':
    pre = Preprocess()
    pre.preprocess_corpus('train.txt','train_TAG.txt',has_tag=True)