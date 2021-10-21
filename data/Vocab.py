from collections import Counter
import numpy as np

class Vocab(object):
    ROOT, PAD, UNK = 0, 1, 2
    def __init__(self, word_counter, rel_counter, max_vocab_size = 1000):
        self._id2word = ['<root>', '<pad>', '<unk>']
        self._id2extword = ['<root>', '<pad>', '<unk>']
        self._id2rel = ['<root>']

        for word, count in word_counter.most_common():
            self._id2word.append(word)
            if len(self._id2word) >= max_vocab_size:
                break

        for rel, count in rel_counter.most_common():
            self._id2rel.append(rel)


        reverse = lambda x: dict(zip(x, range(len(x))))
        self._word2id = reverse(self._id2word)
        if len(self._word2id) != len(self._id2word):
            print("serious bug: words dumplicated, please check!")

        self._rel2id = reverse(self._id2rel)
        if len(self._rel2id) != len(self._id2rel):
            print("serious bug: relations dumplicated, please check!")

        print("relation: ", self._id2rel)
        print("relation size: ", len(self._rel2id))


    def word2id(self, xs):
        if isinstance(xs, list):
            return [self._word2id.get(x, self.UNK) for x in xs]
        return self._word2id.get(xs, self.UNK)

    def id2word(self, xs):
        if isinstance(xs, list):
            return [self._id2word[x] for x in xs]
        return self._id2word[xs]

    def extword2id(self, xs):
        if isinstance(xs, list):
            return [self._extword2id.get(x, self.UNK) for x in xs]
        return self._extword2id.get(xs, self.UNK)

    def id2extword(self, xs):
        if isinstance(xs, list):
            return [self._id2extword[x] for x in xs]
        return self._id2extword[xs]

    def rel2id(self, xs):
        if isinstance(xs, list):
            return [self._rel2id.get(x) for x in xs]
        return self._rel2id.get(xs)

    def id2rel(self, xs):
        if isinstance(xs, list):
            return [self._id2rel[x] for x in xs]
        return self._id2rel[xs]


    @property
    def word_size(self):
        return len(self._id2word)

    @property
    def extword_size(self):
        return len(self._id2extword)

    @property
    def rel_size(self):
        return len(self._id2rel)

    def load_pretrained_embs(self, embfile):
        embedding_dim = -1
        word_count = 0
        with open(embfile, encoding='utf-8') as f:
            for line in f.readlines():
                if word_count < 1:
                    values = line.split()
                    embedding_dim = len(values) - 1
                word_count += 1
        print('Total words: ' + str(word_count) + '\n')
        print('The dim of pretrained embeddings: ' + str(embedding_dim) + '\n')

        index = len(self._id2extword)
        embeddings = np.zeros((word_count + index, embedding_dim))
        with open(embfile, encoding='utf-8') as f:
            for line in f.readlines():
                values = line.split()
                self._id2extword.append(values[0])
                vector = np.array(values[1:], dtype='float64')
                embeddings[self.UNK] += vector
                embeddings[index] = vector
                index += 1

        embeddings[self.UNK] = embeddings[self.UNK] / word_count
        #embeddings = embeddings / np.std(embeddings)

        reverse = lambda x: dict(zip(x, range(len(x))))
        self._extword2id = reverse(self._id2extword)

        if len(self._extword2id) != len(self._id2extword):
            print("serious bug: extern words dumplicated, please check!")

        return embeddings

def create_vocab(instances, max_vocab_size = 1000):
    word_counter = Counter()
    rel_counter = Counter()
    for instance in instances:
        for idx, EDU in enumerate(instance.EDUs):
            if idx == 0: continue ## root
            words = EDU['text'].split(" ")
            for word in words:
                word_counter[word] += 1
        for rel in instance.relations:
            rel_counter[rel['type']] += 1
    return Vocab(word_counter, rel_counter, max_vocab_size)
