"""
File contains utils for evaluation

"""

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report, \
    f1_score


# predictions from softmax are in array like [[0.01,0.6,0.09,0.3],[...]]
# function select the max probability and replace it with 1
# [[0.01,0.6,0.09,0.3],[...]] => [[0,1,0,0],[...]]
def label_probabilities(prob_vectors):
    list = []
    for vector in prob_vectors:
        max_index = np.argmax(vector)
        new_vector = [0] * len(vector)
        new_vector[max_index] = 1
        list.append(new_vector)

    return list



# because keras works with vectorized labels like [[0,0,1,0],[1,0,0,0]]
# we need to to convert to [2,0]
def de_vectorize(label_vectors):
    list = []

    for vector in label_vectors:
        max_index = np.argmax(vector)
        list.append(max_index)

    return list

def compute_measures(y_test, y_pred,classes, print_stats=True):
    precisions = precision_score(y_test, y_pred, average=None)
    recalls = recall_score(y_test, y_pred, average=None)
    f1_scores = f1_score(y_test, y_pred, average=None)

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    micro_f1 = f1_score(y_test, y_pred, average='micro')

    if print_stats:
        # print_str = get_stats_string(accuracy, macro_f1, micro_f1, precisions, recalls, f1_scores,classes)
        str_head, print_str = get_excel_format(accuracy,macro_f1,micro_f1)
        print(str_head)
        print(print_str)


    return accuracy, macro_f1, micro_f1, precisions, recalls, f1_scores

def get_excel_format(accuracy, macro_f1, micro_f1):
    f = "{:<4.4f}"
    str_head = 'Accuracy\tMacro F1\tMicroF1'
    str = f.format(accuracy) + '\t' + f.format(macro_f1) + '\t' + f.format(micro_f1)

    return str_head,str


def get_stats_string(accuracy, macro_f1, micro_f1, precisions, recalls, f1_scores, classes):
    # string = '          ' + ', '.join(classes) + '\n'\
    #       + 'precision:' + ',   '.join(format(x, "2.4f") for x in precisions) + '\n'\
    #       + 'recall:   ' + ',   '.join(format(x, "2.4f") for x in recalls) + '\n' \
    #       + 'f1 score: ' + ',   '.join(format(x, "2.4f") for x in f1_scores) + '\n' \
    string = '----Average----' + '\n' \
          + 'accuracy: %2.4f ' % (accuracy) + '\n' \
          + 'f1 macro score: %2.4f ' % (macro_f1) + '\n' \
          + 'f1 micro score: %2.4f ' % (micro_f1)

    # print('           ', classes)
    # print('precision:', precisions)
    # print('recall:   ', recalls)
    # print('f1 score: ', f1_scores)
    #
    # print('----Average----')
    # print('accuracy ', accuracy)
    # print('f1 macro score: ', macro_f1)
    # print('f1 micro score: ', micro_f1)
    # print_confusion_matrix(y_test,y_pred)

    return string

