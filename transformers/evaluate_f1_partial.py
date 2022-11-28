import os
import sys
import numpy as np
#from sklearn.metrics import f1_score
from f1_score_partial import f1_score


def parse_input(path):
    label_map = {'O': 0, 'B-Object': 1, 'B-Aspect': 2, 'B-Predicate': 3,
                 'I-Aspect': 2, 'I-Predicate': 3, 'I-Object': 1}
    labels = []
    for line in open(path):
        line = line.strip('\n')
        if line == '':
            continue
        else:
            labels.append(label_map[line.split('\t')[1]])
    return labels


# extract labels only
def read_conll(path):
    labels = []
    cur_sent = []
    for line in open(path):
        line = line.strip('\n')
        if line == '':
            labels.append(cur_sent)
            cur_sent = []
            continue
        cur_sent.append(line.split('\t')[-1])
    if len(cur_sent) > 0:
        labels.append(cur_sent)
    return labels


def evaluate_f1(reference, submission, output_filename, pint_mode=False):
    # out = open(output_filename, 'w')
    # compute all the F1-scores
    ref = read_conll(reference)
    hyp = read_conll(submission)
    ind_scores, weights = f1_score(ref, hyp, partial=True, average=None)
    f1_avg = np.average(ind_scores, weights=weights)
    ind_scores_strict, weights_strict = f1_score(ref, hyp, partial=False, average=None)
    f1_avg_strict = np.average(ind_scores_strict, weights=weights_strict)

    # out.write(f'f1_average_strict: {f1_avg_strict:.6f}\n')
    # out.write(f'f1_aspect_strict: {ind_scores_strict[0]:.6f}\n')
    # out.write(f'f1_object_strict: {ind_scores_strict[1]:.6f}\n')
    # out.write(f'f1_predicate_strict: {ind_scores_strict[2]:.6f}\n')
    # out.write(f'f1_average: {f1_avg:.6f}\n')
    # out.write(f'f1_aspect: {ind_scores[0]:.6f}\n')
    # out.write(f'f1_object: {ind_scores[1]:.6f}\n')
    # out.write(f'f1_predicate: {ind_scores[2]:.6f}\n')
    # out.close()
    if pint_mode:
        print(f'f1_average_strict: {f1_avg_strict:.6f}\n')
        print(f'f1_aspect_strict: {ind_scores_strict[0]:.6f}')
        print(f'f1_object_strict: {ind_scores_strict[1]:.6f}')
        print(f'f1_predicate_strict: {ind_scores_strict[2]:.6f}')
        print(f'f1_average: {f1_avg:.6f}')
        print(f'f1_aspect: {ind_scores[0]:.6f}')
        print(f'f1_object: {ind_scores[1]:.6f}')
        print(f'f1_predicate: {ind_scores[2]:.6f}')
    return f1_avg_strict
