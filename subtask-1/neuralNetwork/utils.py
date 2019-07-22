import pandas as pd
from config import *
import pickle


"""
    Utils functions
"""

# create file for evaluation using official script
def create_eval_file(filePath, predictions,labels2dialects):
    # labels = predictions.map(labels2dialects)
    labels = []
    for i,predict in enumerate(predictions):
        labels.append(labels2dialects[predict])

    assert len(labels) == len(predictions)

    with open(filePath,'w',encoding='utf-8') as f:
        for label in labels:
            f.write(label + '\n')

    # print("Labels have been written to:" + filePath)


def append_result(filename, config_str, accuracy, macro_f1, micro_f1):
    f = "{:<4.4f}"
    line = config_str + '\t' + f.format(accuracy) + '\t' + f.format(macro_f1) + '\t' + f.format(micro_f1) + '\n'

    with open(filename,'a') as f:
        f.write(line)


def load_data_s1_names(path, col_names):
    return pd.read_csv(path, names=col_names, encoding='utf-8', sep='\t')

# maybe unite loading
def load_data_s1(path):
    return pd.read_csv(path, names=data_cols_s1, encoding='utf-8', sep='\t')


def save_obj(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def load_obj(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def get_xy(dataset_df, x_col, y_col, title_col, include_title=False):
    # x = dataset_df[x_col].values.astype('U')
    x = dataset_df[x_col]
    if include_title is True:
        x_title = dataset_df[title_col].values.astype('U')
        for i, (title, content) in enumerate(zip(x_title, x)):
            # check if the title is not empty
            if title.strip():
                x[i] = title + ' . ' + content

    y = dataset_df[y_col]

    return x, y







