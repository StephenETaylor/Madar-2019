import json
import os

import tensorflow as tf
from numpy.random import seed
import sklearn.metrics

from print_cm import plot_cm, print_cm

seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

# without this I was not able to run it from command line
import sys
print(sys.path)
sys.path.append('../')

from tensorflow.python import keras
from config import BASE_PATH, DATASET, label_col, TRAIN_MOD_DIR, feature_cols_s1, data_cols_s1, data_test_cols_s1
from eval import compute_measures, get_excel_format, label_probabilities, de_vectorize
from nn.nn_model import build_and_run
from nn.nn_utils import save_model, get_actual_time, load_model_keras
from nn.sequence_utils import init_char_ngram_tokenizers
from utils import create_eval_file, append_result, load_data_s1_names, get_xy
from keras.utils import to_categorical
from tensorflow.python.client import device_lib
from keras import backend as K
from keras.backend.tensorflow_backend import set_session


Dialects = 'ALE ALG ALX AMM ASW BAG BAS BEI BEN CAI DAM DOH FES JED JER KHA MOS MSA MUS RAB RIY SAL SAN SFX TRI TUN'.split()
Dialects_6 = 'MSA RAB TUN BEI DOH CAI'.split()

# create mapping for dialects
dialect2label = {}
label2dialect = {}

NUMBER_CPUS = 6

# todo udelat gitignore abych tam nenahral soubory
# todo kouknout na ten model jak se trenuje
# dat tam ten final model

def predict_test(lang_feature_names, data_test_cols_s1, output_file_name, data_key,y_test=None,labels=None):
    print('base path',BASE_PATH)
    path = os.path.join(BASE_PATH,'trained_models/usable/B_26__BS-41600_EC-800_Att-True_LR-0-01000_Opt-Adam_Final-True_Layers-2_Cells-400_Lang_Dropout-0-0_Test-#DEVS1_2019-05-08_17-15_Acc-0.6608.h5')
    model = load_model_keras(path)

    X_test = load_data_test(lang_feature_names, data_test_cols_s1, data_key)
    y_pred = model.predict(X_test)

    y_pred = label_probabilities(y_pred)
    y_pred = de_vectorize(y_pred)
    y_pred_labels = [label2dialect[i] for i in y_pred]

    with open(output_file_name, mode='w', encoding='utf-8') as f:
        for i in y_pred_labels:
            f.write(i + '\n')

    if y_test is not None:
        y_test = de_vectorize(y_test)
        cm = sklearn.metrics.confusion_matrix(y_test, y_pred)
        print_cm(cm, labels=labels)
        plot_cm(cm, labels=labels)
        #
        # percentage_matrix = 100 * cm / cm.sum(axis=1).astype(float)
        # plot_cm(percentage_matrix, labels=labels, fmt='.2f')


def main():
    test = True

    config_file = "./model_configs/ngram_base_line.config"

    with open(os.path.join(BASE_PATH, config_file)) as configfile:
        config = json.load(configfile)

    # load data
    if config['task'] == 's1-26':
        dial = Dialects
        lang_feature_names = feature_cols_s1
        col_names = data_cols_s1
        config['out_len'] = 26
    else:
        raise Exception("Task not specified")

    ngram_voc_max_size = config['ngram_voc_max_size']
    x_train, y_train, x_test, y_test, x_train_lang_features, x_test_lang_features, \
    x_train_word_lang_features, x_test_word_lang_features = \
        load_data(config, dial, lang_feature_names, col_names)

    if test:
        predict_test(lang_feature_names,data_cols_s1,
                     os.path.join(BASE_PATH, 'test-output/output-dev.txt'), 'DEV_DATA_S1', y_test=y_test, labels=dial)
        # predict_test(lang_feature_names,data_test_cols_s1,os.path.join(BASE_PATH, 'test-output/output-test.txt'),'TEST_DATA_S1')

    else:

        vaporizers_dict = init_char_ngram_tokenizers(x_train, ngram_range=(config['ngram_min'],
                                                                            config['ngram_max']),
                                                                            max_features=ngram_voc_max_size)

        if config['use_word_features'] is True:
            config['lang_features_len'] = 2 * config['lang_features_len']

        print("#########Time start:", get_actual_time(), "########")
        run_nn_experiments(config, x_train, y_train, x_test, y_test, vaporizers_dict,
                           x_train_lang_features, x_test_lang_features, x_train_word_lang_features,x_test_word_lang_features)
        print("#########Time end:", get_actual_time(), "########")


def run_nn_experiments(config, x_train, y_train, x_test, y_test, vectorizers_dict,
                       x_train_lang_features, x_test_lang_features, x_train_word_lang_features,x_test_word_lang_features):

    y_pred, y_test, model, tensorboard_log_dir, train_test_time, callback_list =\
        build_and_run(x_train, y_train, x_test, y_test, vectorizers_dict
                      , config, x_train_lang_features, x_test_lang_features, x_train_word_lang_features,x_test_word_lang_features)

    # compute measures
    accuracy, macro_f1, micro_f1, precisions, recalls, f1_scores = compute_measures(y_test, y_pred,None)

    # get excel results and print
    str_head, final_ret = get_excel_format(accuracy, macro_f1, micro_f1)
    print(str_head + '\n' + final_ret)
    print("train and test time: ", train_test_time)

    embeddings_name = config['embeddings_file']

    # save model
    file_name_model = save_model(model, config, embeddings_name, tensorboard_log_dir, train_test_time,
               accuracy, macro_f1, micro_f1, precisions, recalls, f1_scores,callback_list)

    res_file_name = os.path.join(TRAIN_MOD_DIR,'allres.txt')

    append_result(res_file_name, file_name_model, accuracy, macro_f1, micro_f1)

    # run_nn_model_ngrams(x_train, y_train, x_test, y_test, vectorizers_dict,config)



def load_data_test(lang_feature_names, col_names, dataset_key_path):
    data_test_df = load_data_s1_names(DATASET.path[dataset_key_path], col_names)
    data_test_df['sent'] = data_test_df['sent'].str.strip()

    x_test_lang_features = data_test_df[lang_feature_names]

    return x_test_lang_features

def load_data(config, dialects, lang_feature_names, col_names):
    data_train_df_s1 = load_data_s1_names(DATASET.path[config['train_data']],col_names)
    data_train_df_s1['sent'] = data_train_df_s1['sent'].str.strip()

    data_dev_df_s1 = load_data_s1_names(DATASET.path[config['test_data']],col_names)
    data_dev_df_s1['sent'] = data_dev_df_s1['sent'].str.strip()

    for i, dial in enumerate(dialects):
        dialect2label[dial] = i
        label2dialect[i] = dial

    # map text label to numbers
    data_train_df_s1[label_col] = data_train_df_s1[label_col].map(dialect2label)
    data_dev_df_s1[label_col] = data_dev_df_s1[label_col].map(dialect2label)

    # create file with dev gold
    create_eval_file(DATASET.path[config['gold']], data_dev_df_s1[label_col], label2dialect)

    x_train_lang_features, x_test_lang_features, x_train_word_lang_features, x_test_word_lang_features = None,None,None,None
    if lang_feature_names is not None:
        x_train_lang_features = data_train_df_s1[lang_feature_names]
        x_test_lang_features = data_dev_df_s1[lang_feature_names]

        x_train_word_lang_features,x_test_word_lang_features =\
            extract_word_features(col_names,lang_feature_names,config)

    x_train, y_train = get_xy(data_train_df_s1, 'sent', label_col, None)
    x_test, y_test = get_xy(data_dev_df_s1, 'sent', label_col, None)


    # x_train, y_train, x_train_lang_features = shuffle(x_train, y_train, x_train_lang_features)

    y_train = to_categorical(y_train, num_classes=len(dialects))
    y_test = to_categorical(y_test, num_classes=len(dialects))

    print('Loaded ' + str(len(x_train)) + " train documents")
    print('Loaded ' + str(len(x_test)) + " test documents")


    return x_train, y_train, x_test, y_test,\
           x_train_lang_features, x_test_lang_features, x_train_word_lang_features,x_test_word_lang_features

def extract_word_features(col_names, lang_feature_names,config):
    train_data_path = config['train_data'] + "_WORDS"
    test_data_path = config['test_data'] + "_WORDS"
    data_train_df_s1 = load_data_s1_names(DATASET.path[train_data_path], col_names)
    data_dev_df_s1 = load_data_s1_names(DATASET.path[test_data_path], col_names)

    x_train_word_lang_features = data_train_df_s1[lang_feature_names]
    x_test_word_lang_features = data_dev_df_s1[lang_feature_names]

    return x_train_word_lang_features,x_test_word_lang_features


if __name__ == '__main__':
    print(device_lib.list_local_devices())
    config = tf.ConfigProto(intra_op_parallelism_threads=NUMBER_CPUS, inter_op_parallelism_threads=NUMBER_CPUS)
    session = tf.Session(config=config)
    K.tensorflow_backend.set_session(session)

    print('Keras version:' + keras.__version__)
    print('Tensor flow version:' + str(tf.__version__))

    main()