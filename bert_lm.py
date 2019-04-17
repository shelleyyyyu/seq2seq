from pytorch_pretrained_bert import BertTokenizer, BertForMaskedLM
#from bert.modeling import BertForMaskedLM
#from bert.tokenization import BertTokenizer

import torch
import math

class ModelEvaluator:

    def __init__(self):
        self.bertMaskedLM = BertForMaskedLM.from_pretrained('bert-base-uncased', cache_dir='./bert_model'.decode('unicode-escape'))
        self.bertMaskedLM.eval()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir='./bert_model'.decode('unicode-escape'))


    def get_score(self, sentence):
        tokenize_input = self.tokenizer.tokenize(sentence)
        tensor_input = torch.tensor([self.tokenizer.convert_tokens_to_ids(tokenize_input)])
        predictions = self.bertMaskedLM(tensor_input)
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(predictions.squeeze(), tensor_input.squeeze()).data
        return math.exp(loss)


    def getNewTrainingPair(self, original_result_list, result_list, ground_truth_list):
        original_result_list = ["They waited close to an hour .","They were finally able to go inside .","And Kate stepped on a step and broke her heel .","Her night was already ruined .","I was trying to watch my diet ."]
        result_list = ["They waited close to an hour .","They were finally able to go inside .","And Kate stepped on a step and broke her heel .","Her night was already ruined .","I was trying to watch my diet ."]
        ground_truth_list = ["They waited close to an hour .","They were finally able to go inside .","And Kate stepped on a step and broke her heel .","Her night was already ruined .","I was trying to watch my diet ."]

        np_array = []
        for index in range(len(original_result_list)):
            for i in range(4):
                np_array.append([original_result_list[index][i], result_list[index][i], ground_truth_list[index][i]])

        print np_array[0]

        for j in range(len(np_array)):
            print(j, [self.get_score(i) for i in np_array[j]])
