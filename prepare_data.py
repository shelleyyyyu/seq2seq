import csv
import pickle as pkl
import nltk

def load_csv(filename):
    with open(filename, 'rb') as File:
        stories = []
        lines = csv.reader(File)
        for line in lines:
            stories.append(line[1:])
    return stories[1:]

def formatString(datastr):
    datastr = nltk.WordPunctTokenizer().tokenize(datastr.lower())
    newdatastr = ''
    for index, d in enumerate(datastr):
        newdatastr = newdatastr + d + " "
    return newdatastr

def formatTitle(datastr):
    datastr = nltk.WordPunctTokenizer().tokenize(datastr.lower())
    newdatastr = ''
    for index, d in enumerate(datastr):
        newdatastr = newdatastr + d + ""
    return newdatastr


def prepare_single(data):
    Data = dict()

    dataLen = len(data)
    train_lens = int(dataLen * 0.8)
    val_lens = int(dataLen * 0.1)
    train_data = data[:train_lens]
    valid_data = data[train_lens:train_lens+val_lens]
    test_data = data[train_lens+val_lens:]


    with open("./data/train.txt", "wr") as f:
        for data in train_data:
            datastr = "[{}]{}<split>{}<split>{}<split>{}<split>{}<split>".format(formatTitle(data[0]), formatString(data[1]), formatString(data[2]), formatString(data[3]), formatString(data[4]), formatString(data[5]))
            f.write(datastr+"\n")
    f.close()

    with open("./data/valid.txt", "wr") as f:
        for data in valid_data:
            datastr = "[{}]{}<split>{}<split>{}<split>{}<split>{}<split>".format(formatTitle(data[0]), formatString(data[1]), formatString(data[2]), formatString(data[3]), formatString(data[4]), formatString(data[5]))
            f.write(datastr+"\n")
    f.close()

    with open("./data/test.txt", "wr") as f:
        for data in test_data:
            datastr = "[{}]{}<split>{}<split>{}<split>{}<split>{}<split>".format(formatTitle(data[0]), formatString(data[1]), formatString(data[2]), formatString(data[3]), formatString(data[4]), formatString(data[5]))
            f.write(datastr+"\n")
    f.close()






data = load_csv('./data/rocStories.csv')
prepare_single(data)

