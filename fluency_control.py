import math
from language_model_tools.language_model import Language_Model
from six.moves import cPickle
import tensorflow as tf
import os
import time
#https://github.com/abolibibelot/nlp-playground/blob/master/perplexity.py

class ModelEvaluator:
    def get_language_model_checkpoint(self):
        lm_graph = tf.Graph()
        lm_sess = tf.Session(graph=lm_graph)
        lm_save_path = "language_model_save"

        # setup Language model
        with open(os.path.join(lm_save_path, 'words_vocab.pkl'), 'rb') as f:
            lm_words, vocab = cPickle.load(f)


        # retrieve language model checkpoint and save to the lm_sess
        with lm_sess.as_default():
            with lm_graph.as_default():
                with open(os.path.join(lm_save_path, 'config.pkl'), 'rb') as f:
                    saved_args = cPickle.load(f)
                    lm_model = Language_Model(saved_args, True)

                lm_vocab = vocab
                lm_ckpt = tf.train.get_checkpoint_state(lm_save_path)
                if lm_ckpt and lm_ckpt.model_checkpoint_path:
                    lm_model.saver.restore(lm_sess, lm_ckpt.model_checkpoint_path)
        return lm_graph, lm_sess, lm_model, lm_vocab, lm_ckpt, lm_words

    def prob(self, lm_graph, lm_sess, lm_model, lm_vocab, sentence):

        with lm_sess.as_default():
            with lm_graph.as_default():
                #print "===========sentence========"
                #print sentence
                items = []
                for index in range(len(sentence)):
                    if index == 0:
                        word = sentence[index]
                        items.append(['', word])
                    else:
                        prime = sentence[:index]
                        primeStr = " ".join(prime)
                        word = sentence[index]
                        items.append([primeStr, word])
                #print "===========items========"
                #print items
                '''acc = 1
                for item in items:
                    prob = lm_model.getProb(lm_sess, item[0], item[1], lm_vocab)

                    print "===========item[0]========"
                    print item[0]

                    print "===========item[1]========"
                    print item[1]

                    print "===========prob========"
                    print prob
                    acc = acc * prob
                    print "===========acc========"
                    print acc'''

                acc = reduce(lambda acc,x:acc * lm_model.getProb(lm_sess, x[0], x[1], lm_vocab),items,1)
                return acc

    def perplexity(self, lm_graph, lm_sess, lm_model, lm_vocab, sentence):
        word_count = len(sentence)
        prob = self.prob(lm_graph, lm_sess, lm_model, lm_vocab, sentence)
        if prob != 0.0:
            prob = math.log(prob, 2)
            logpart = (- 1.0 * prob) / word_count
            return float(1 / (1+logpart))
        else:
            return 0.0

    def getNewTrainingPair(self, original_result_list, result_list, ground_truth_list):
        new_training_data = []
        # get language model checkpoint
        count= 0

        #for list_index in range(min(len(result_list), len(ground_truth_list))):
        for list_index in range(len(result_list)):
            print len(result_list[list_index])
            for index in range(4):
                lm_graph, lm_sess, lm_model, lm_vocab, lm_ckpt, lm_words = self.get_language_model_checkpoint()
                original_sentence = original_result_list[list_index][index]
                sentence = result_list[list_index][index]
                ground_truth_sentence = ground_truth_list[list_index][index]
                s1 = self.perplexity(lm_graph, lm_sess, lm_model, lm_vocab, original_sentence.split(" "))
                s2 = self.perplexity(lm_graph, lm_sess, lm_model, lm_vocab, sentence.split(" "))
                s3 = self.perplexity(lm_graph, lm_sess, lm_model, lm_vocab, ground_truth_sentence.split(" "))
                print ("s1: %s, s2: %s, s3: %s"%(original_sentence,sentence,ground_truth_sentence))
                print ("s1: %f, s2: %f, s3: %f"%(s1,s2,s3))
                if s1 < s2 and s2 < s3:
                    count += 1

        print "end"
        print count



    def getNewTrainingPair(self, original_result_list, result_list, ground_truth_list):


        #for list_index in range(min(len(result_list), len(ground_truth_list))):
        for list_index in range(len(result_list)):
            for index in range(4):
                original_sentence = original_result_list[list_index][index]
                sentence = result_list[list_index][index]
                ground_truth_sentence = ground_truth_list[list_index][index]

