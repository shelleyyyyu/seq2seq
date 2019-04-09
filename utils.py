from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
import re
import collections
import pickle
import numpy as np
from gensim.models.keyedvectors import KeyedVectors
from gensim.test.utils import get_tmpfile
from gensim.scripts.glove2word2vec import glove2word2vec


train_path = "data/train.txt"
test_path = "data/test.txt"


def clean_str(sentence):
    #sentence = re.sub("[#.]+", " ", sentence)
    sentence = sentence.replace("[", "")
    sentence = sentence.replace("]", " ")
    sentence = sentence.replace("<split>", " ")
    return sentence


def get_text_list(data_path, toy):
    with open(data_path, "r") as f:
        if not toy:
            return [clean_str(x.strip()) for x in f.readlines()]
        else:
            return [clean_str(x.strip()) for x in f.readlines()][:50000]



def get_train_list(data_path):
    with open(data_path, "r") as f:
        story_list = []
        for line in f.readlines():
            total_story = []
            line_array = line.split("]")
            title = line_array[0][1:]
            story = line_array[1].split("<split>")[:-1]

            total_story.append(title)
            total_story.extend(story)
            story_list.append(total_story)
        return story_list


def get_test_list(data_path):
    with open(data_path, "r") as f:
        story_list = []
        title_list = []
        for line in f.readlines():
            total_story = []
            line_array = line.split("]")
            title = line_array[0][1:]
            story = line_array[1].split("<split>")[:-1]

            total_story.append(title)
            total_story.extend(story[:-1])
            story_list.append(total_story)
            title_list.append(title)
        print ('================get_test_list=================')
        print title_list[0]
        print story_list[0]
        return title_list, story_list



def build_dict(step, toy=False):
    if step == "train":
        train_story_list = get_text_list(train_path, toy)
        words = list()
        for sentence in train_story_list :
            for word in word_tokenize(sentence):
                words.append(word)

        word_counter = collections.Counter(words).most_common()
        word_dict = dict()
        word_dict["<padding>"] = 0
        word_dict["<unk>"] = 1
        word_dict["<s>"] = 2
        word_dict["</s>"] = 3
        for word, _ in word_counter:
            word_dict[word] = len(word_dict)

        with open("word_dict.pickle", "wb") as f:
            pickle.dump(word_dict, f)

    elif step == "test":
        with open("word_dict.pickle", "rb") as f:
            word_dict = pickle.load(f)

    reversed_dict = dict(zip(word_dict.values(), word_dict.keys()))

    article_max_len = 30
    summary_max_len = 30

    return word_dict, reversed_dict, article_max_len, summary_max_len

def build_train_dataset(word_dict, article_max_len, summary_max_len):
    story_list = get_train_list(train_path)
    x = []
    y = []
    for story in story_list:
        for index in range(len(story)-1):
            #if index == 0:
            x.append(story[index])
            y.append(story[index+1])

    x = [word_tokenize(d) for d in x]
    x = [[word_dict.get(w, word_dict["<unk>"]) for w in d] for d in x]
    x = [d[:article_max_len] for d in x]
    x = [d + (article_max_len - len(d)) * [word_dict["<padding>"]] for d in x]

    y = [word_tokenize(d) for d in y]
    y = [[word_dict.get(w, word_dict["<unk>"]) for w in d] for d in y]
    y = [d[:(summary_max_len - 1)] for d in y]

    return x, y


def build_test_dataset(word_dict, article_max_len):
    title_list, story_list = get_test_list(test_path)
    s = [[word_tokenize(d) for d in story]for story in story_list]
    s = [[[word_dict.get(w, word_dict["<unk>"]) for w in d] for d in x]for x in s]
    s = [[d[:article_max_len] for d in x] for x in s]
    s = [[d + (article_max_len - len(d)) * [word_dict["<padding>"]] for d in x] for x in s]
    return title_list, s




def batch_iter(inputs, outputs, batch_size, num_epochs):
    inputs = np.array(inputs)
    outputs = np.array(outputs)

    num_batches_per_epoch = (len(inputs) - 1) // batch_size + 1
    for epoch in range(num_epochs):
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, len(inputs))
            yield inputs[start_index:end_index], outputs[start_index:end_index]


def test_batch_iter(inputs, outputs, batch_size, num_epochs):
    inputs = np.array(inputs)
    outputs = np.array(outputs)

    num_batches_per_epoch = (len(inputs) - 1) // batch_size + 1
    for epoch in range(num_epochs):
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, len(inputs))
            yield inputs[start_index:end_index], outputs[start_index:end_index]


def get_init_embedding(reversed_dict, embedding_size):
    glove_file = "glove/glove.42B.300d.txt"
    word2vec_file = get_tmpfile("word2vec_format.vec")
    glove2word2vec(glove_file, word2vec_file)
    print("Loading Glove vectors...")
    word_vectors = KeyedVectors.load_word2vec_format(word2vec_file)

    word_vec_list = list()
    for _, word in sorted(reversed_dict.items()):
        try:
            word_vec = word_vectors.word_vec(word)
        except KeyError:
            word_vec = np.zeros([embedding_size], dtype=np.float32)

        word_vec_list.append(word_vec)

    # Assign random vector to <s>, </s> token
    word_vec_list[2] = np.random.normal(0, 1, embedding_size)
    word_vec_list[3] = np.random.normal(0, 1, embedding_size)

    return np.array(word_vec_list)
