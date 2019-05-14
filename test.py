import tensorflow as tf
import pickle
from model import Model
from utils import build_dict, build_test_dataset, test_batch_iter
import os


with open("args.pickle", "rb") as f:
    args = pickle.load(f)

print("Loading dictionary...")
word_dict, reversed_dict, article_max_len, summary_max_len = build_dict("test", False)
print("Loading test dataset...")
title_list, test_x = build_test_dataset(word_dict, article_max_len)
test_x_len = [len([y for y in x if y != 0]) for x in test_x]

with tf.Session() as sess:
    print("Loading saved model...")
    model = Model(reversed_dict, article_max_len, summary_max_len, args, forward_only=True)
    saver = tf.train.Saver(tf.global_variables())
    ckpt = tf.train.get_checkpoint_state("./saved_model/")
    saver.restore(sess, ckpt.model_checkpoint_path)

    story_test_result_list = []
    if args.use_atten:
        print ("Using Attention")
    else:
        print ("Not Using Attention")

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
            for line in prediction_output:
                summary = list()
                for word in line:
                    if word == "</s>":
                        break
                    if word not in summary:
                        summary.append(word)
                predict_story = predict_story+" ".join(summary)+"<split>"


        predict_story = "[{}]{}".format(title_list[index], predict_story)
        story_test_result_list.append(predict_story)



    if not os.path.exists("result"):
            os.mkdir("result")
    with open("result/test.txt", "wr") as f:
        for story in story_test_result_list:
            f.write(story+"\n")
    print('Summaries are saved to "test.txt"...')
