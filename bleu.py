# -*- coding: utf-8 -*-
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction
import sys
import re


def cal_corpus_bleu(refs, hyps, k):
    if k == 1:
        w = (1, 0, 0, 0)
    elif k == 2:
        w = (0.5, 0.5, 0, 0)
    elif k == 3:
        w = (0.33, 0.33, 0.33, 0)
    elif k == 4:
        w = (0.25, 0.25, 0.25, 0.25)
    else:
        return
    bleu_k = corpus_bleu(refs, hyps, weights = w)
    bleu_k = bleu_k * 100
    return bleu_k

def main(argv):
    ref_file = str(argv[1])
    hyp_file = str(argv[2])
    refs = []
    hyps = []
    with open(ref_file, "r") as f:
        ref_lines = f.readlines()
        for line in ref_lines:
            refs.append([line.strip().split()])
    with open(hyp_file, "r") as f:
        hyp_lines = f.readlines()
        for line in hyp_lines:
            hyps.append(line.strip().split())
    bleu1 = cal_corpus_bleu(refs, hyps, 1)
    bleu2 = cal_corpus_bleu(refs, hyps, 2)
    bleu3 = cal_corpus_bleu(refs, hyps, 3)
    bleu4 = cal_corpus_bleu(refs, hyps, 4)
    print("bleu score 1 ~ 4 ", bleu1, bleu2, bleu3, bleu4)

if __name__ == '__main__':
    main(sys.argv)
