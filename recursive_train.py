import time
#start = time.perf_counter()
import tensorflow as tf
import argparse
import pickle
import os
from model import Model
from utils import build_dict, build_train_dataset, batch_iter, build_test_dataset, build_train_edit_dataset, test_batch_iter

# Uncomment next 2 lines to suppress error and Tensorflow info verbosity. Or change logging levels
# tf.logging.set_verbosity(tf.logging.FATAL)
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def add_arguments(parser):
    parser.add_argument("--num_hidden", type=int, default=150, help="Network size.")
    parser.add_argument("--num_layers", type=int, default=2, help="Network depth.")
    parser.add_argument("--beam_width", type=int, default=1, help="Beam width for beam search decoder.")
    parser.add_argument("--glove", action="store_true", help="Use glove as initial word embedding.")
    parser.add_argument("--embedding_size", type=int, default=300, help="Word embedding size.")
    parser.add_argument("--recursive_count", type=int, default=3, help="Word embedding size.")

    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of epochs.")
    parser.add_argument("--keep_prob", type=float, default=0.8, help="Dropout keep prob.")

    parser.add_argument("--toy", action="store_true", help="Use only 50K samples of data")
    parser.add_argument("--use_atten", action="store_true", help="Use attention")

    parser.add_argument("--with_model", action="store_true", help="Continue from previously saved model")


def train(train_path, train_gt_path):
    print("Loading training dataset...")
    train_x, train_y = build_train_dataset(train_path, train_gt_path, word_dict, article_max_len, summary_max_len)
    with tf.Session() as sess:
        model = Model(reversed_dict, article_max_len, summary_max_len, args)
        if args.use_atten:
            print ("Using Attention")
        else:
            print ("Not Using Attention")
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())
        if 'old_model_checkpoint_path' in globals():
            print("Continuing from previous trained model:" , old_model_checkpoint_path , "...")
            saver.restore(sess, old_model_checkpoint_path )

        batches = batch_iter(train_x, train_y, args.batch_size, args.num_epochs)
        num_batches_per_epoch = (len(train_x) - 1) // args.batch_size + 1

        print("\nIteration starts.")
        print("Number of batches per epoch :", num_batches_per_epoch)
        for batch_x, batch_y in batches:
            batch_x_len = list(map(lambda x: len([y for y in x if y != 0]), batch_x))
            batch_decoder_input = list(map(lambda x: [word_dict["<s>"]] + list(x), batch_y))
            batch_decoder_len = list(map(lambda x: len([y for y in x if y != 0]), batch_decoder_input))
            batch_decoder_output = list(map(lambda x: list(x) + [word_dict["</s>"]], batch_y))

            batch_decoder_input = list(
                map(lambda d: d + (summary_max_len - len(d)) * [word_dict["<padding>"]], batch_decoder_input))
            batch_decoder_output = list(
                map(lambda d: d + (summary_max_len - len(d)) * [word_dict["<padding>"]], batch_decoder_output))

            train_feed_dict = {
                model.batch_size: len(batch_x),
                model.X: batch_x,
                model.X_len: batch_x_len,
                model.decoder_input: batch_decoder_input,
                model.decoder_len: batch_decoder_len,
                model.decoder_target: batch_decoder_output
            }

            _, step, loss = sess.run([model.update, model.global_step, model.loss], feed_dict=train_feed_dict)

            if step % 1000 == 0:
                print("step {0}: loss = {1}".format(step, loss))

            if step % num_batches_per_epoch == 0:
                saver.save(sess, "./saved_model/model.ckpt", global_step=step)
                print(" Epoch {0}: Model is saved.".format(step // num_batches_per_epoch))


def test(sess, args, model, test_title_list, test_x, reversed_dict, save_filename):
    story_test_result_list = []
    story_test_result_array = []
    for index in range(len(test_x)):
        inputs = test_x[index]
        batches = test_batch_iter(inputs, [0] * len(test_x), args.batch_size, 1)

        result = []
        for batch_x, _ in batches:
            batch_x_len = [len([y for y in x if y != 0]) for x in batch_x]

            test_feed_dict = {
                model.batch_size: len(batch_x),
                model.X: batch_x,
                model.X_len: batch_x_len,
            }

            prediction = sess.run(model.prediction, feed_dict=test_feed_dict)
            prediction_output = [[reversed_dict[y] for y in x] for x in prediction[:, 0, :]]
            predict_story = ""
            predict_story_array = []
            for line in prediction_output:
                summary = list()
                for word in line:
                    if word == "</s>":
                        break
                    if word not in summary:
                        summary.append(word)
                predict_story = predict_story+" ".join(summary)+"<split>"
                predict_story_array.append(" ".join(summary))


        predict_story = "[{}]{}".format(test_title_list[index], predict_story)
        story_test_result_list.append(predict_story)
        story_test_result_array.append(predict_story_array)


    if not os.path.exists("result_edit"):
        os.mkdir("result_edit")
    filename = "result_edit/{}".format(save_filename)
    with open(filename, "wr") as f:
        for story in story_test_result_list:
            f.write(story+"\n")
    print('Summaries are saved to "{}"...').format(save_filename)




parser = argparse.ArgumentParser()
add_arguments(parser)
args = parser.parse_args()
with open("args.pickle", "wb") as f:
    pickle.dump(args, f)

print("Building dictionary...")
word_dict, reversed_dict, article_max_len, summary_max_len = build_dict("train", args.toy)
for index in range(args.recursive_count):
    tf.reset_default_graph()
    if not os.path.exists("saved_model"):
        os.mkdir("saved_model")
    else:
        if index != 0:
        #if args.with_model:
            old_model_checkpoint_path = open('saved_model/checkpoint', 'r')
            old_model_checkpoint_path = "".join(["saved_model/", old_model_checkpoint_path.read().splitlines()[0].split('"')[1] ])


    if index == 0:
        train_path = "result/train.txt"
        train_gt_path = "result/train_ground_truth.txt"
    else:
        train_path = "result_edit/train_{}.txt".format(index-1)
        train_gt_path = "result/train_ground_truth.txt"

    print "[[[[[[[[[[Start Training]]]]]]]]]]]]]"
    train(train_path, train_gt_path)

    tf.reset_default_graph()


    print "[[[[[[[[[[Start Testing]]]]]]]]]]]]]"
    with open("args.pickle", "rb") as f:
        args = pickle.load(f)

    print("Loading dictionary...")
    word_dict, reversed_dict, article_max_len, summary_max_len = build_dict("test", args.toy)

    with tf.Session() as sess:
        print("Loading saved model...")
        model = Model(reversed_dict, article_max_len, summary_max_len, args, forward_only=True)
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state("./saved_model/")
        saver.restore(sess, ckpt.model_checkpoint_path)

        if args.use_atten:
            print ("Using Attention")
        else:
            print ("Not Using Attention")


        print("Loading test dataset...")

        if index == 0:
            path = "./result/test.txt"
        else:
            path = "./result_edit/test_{}.txt".format(index-1)

        test_title_list, test_x = build_test_dataset(path, word_dict, article_max_len)


        if index == 0:
            path = "./result/train.txt"
        else:
            path = "./result_edit/train_{}.txt".format(index-1)
        train_title_list, train_x = build_train_edit_dataset(path, word_dict, article_max_len)

        test_x_len = [len([y for y in x if y != 0]) for x in test_x]
        train_x_len = [len([y for y in x if y != 0]) for x in train_x]

        save_filename = "test_{}.txt".format(index)

        test(sess, args, model, test_title_list, test_x, reversed_dict, save_filename)

        save_filename = "train_{}.txt".format(index)
        test(sess, args, model, train_title_list, train_x, reversed_dict, save_filename)

