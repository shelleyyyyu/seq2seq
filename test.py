import tensorflow as tf
import pickle
from model import Model
from utils import build_dict, build_test_dataset, test_batch_iter, build_test_gt_dataset, build_train_edit_dataset
import os
from fluency_control import ModelEvaluator


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

    path = "./result/test.txt"

    test_title_list, test_x = build_test_dataset(path, word_dict, article_max_len)

    path = "./result/train.txt"
    train_title_list, train_x = build_train_edit_dataset(path, word_dict, article_max_len)

    test_x_len = [len([y for y in x if y != 0]) for x in test_x]
    train_x_len = [len([y for y in x if y != 0]) for x in train_x]

    save_filename = "test.txt"
    test(sess, args, model, test_title_list, test_x, reversed_dict, save_filename)
    save_filename = "train.txt"
    test(sess, args, model, train_title_list, train_x, reversed_dict, save_filename)



    '''filter the results and create another version of dataset for training
    test_path = "result/train.txt"
    test_result_path = "result_edit/train_0.txt"
    test_gt_path = "result/train_ground_truth.txt"
    test_original_list = build_test_gt_dataset(test_path)
    story_test_result_array = build_test_gt_dataset(test_result_path)
    test_gt_list = build_test_gt_dataset(test_gt_path)'''

