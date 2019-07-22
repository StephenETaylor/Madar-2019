import os
import shutil
import datetime
import json
from contextlib import redirect_stdout
import pickle

from keras.engine.saving import load_model

from config import TRAIN_MOD_DIR
from eval import get_stats_string, get_excel_format


def generate_file_name(config, embeddings_name, accuracy=None, epochs=None):
    time = get_actual_time()
    use_word_emb = config['use_word_emb']
    use_ngram = config['use_ngram']
    use_lang_features = config['use_lang_features']

    if epochs is None:
        epochs = config['epoch_count']

    name_file = config['model_name'] + "_" \
    + "_BS-" + str(config['batch_size']) \
    + "_EC-" + str(epochs) \
    + "_Att-" + str(config['attention']) \
    + "_LR-%.5f" % (config['lr']) \
    + "_Opt-" + config['optimizer'] \
    + "_Final-" + str(config['final_layer'])

    if use_word_emb is True:
       name_file += embeddings_name + "_EDim" + str(config['embeddings_dimension'])
       name_file += "_Cells-" + str(config['cells'])

    if use_ngram is True:
        name_file += "_Ngram-" + str(config['ngram_min']) + "-" + str(config['ngram_max'])
        name_file += "_Cells_ngram-" + str(config['ngram_cells'])

    if use_lang_features is True:
        name_file += "_Layers-" + str(config['num_lang_dense_layers'])
        name_file += "_Cells-" + str(config['dense_lang_layers_cells'])
        name_file += "_Lang_Dropout-" + str(config['dense_lang_dropout'])

        if config['use_word_features'] is True:
            name_file += "_Word-feature-" + str(config['use_word_features'])


    name_file += "_Test-" + "#" + config['test_data']
    name_file += "_" + time


    # name_file = config['model_name'] + "_" \
    #             + embeddings_name + "_EDim" + str(config['embeddings_dimension']) \
    #             + "_BS-" + str(config['batch_size']) \
    #             + "_EC-" + str(config['epoch_count']) \
    #             + "_Att-" + str(config['attention']) \
    #             + "_LR-%.5f" % (config['lr']) \
    #             + "_Opt-" + config['optimizer'] \
    #             + "_Cells-" + str(config['cells']) \
    #             + "_Ngram-"+ str(config['ngram_min']) + "-" + str(config['ngram_max']) \
    #             + "_Cells_ngram-" + str(config['ngram_cells']) \
    #             + "_Train-" + "#" + config['train_data'] \
    #             + "_Test-" + "#" + config['test_data'] \
    #             + "_" + time
    #

    name_file = name_file.replace('.','-')
    if accuracy is not None:
        name_file = name_file + "_Acc-%.4f" % (accuracy)

    return name_file



def get_actual_time():
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d_%H.%M")

def load_model_keras(model_path):
    return load_model(model_path)


def save_model(model, config, embeddings_name, tensorboard_log_dir, train_test_time,
               accuracy, macro_f1, micro_f1, precisions, recalls, f1_scores,callback_list):
    name_folder = os.path.join(TRAIN_MOD_DIR,config['model_name'])

    if os.path.exists(name_folder) is not True:
        os.makedirs(name_folder)

    epochs = None
    if config['use_early_stopping'] is True:
        es = callback_list[1]
        epochs = es.stopped_epoch

    # generate file name
    name_file = generate_file_name(config, embeddings_name, accuracy=accuracy,epochs=epochs)
    print("File name:", name_file)

    # copy tensorboard log file to new directory
    copy_files(tensorboard_log_dir, name_folder)

    # save trained model
    model_file_name = name_file + ".h5"
    model.save(os.path.join(name_folder,model_file_name))

    # assing file name
    config['model_file_name'] = model_file_name

    # dump used config
    with open(os.path.join(name_folder, name_file + ".config"), 'w') as outfile:
        json.dump(config, outfile, indent=4)

    # dump model summary
    with open(os.path.join(name_folder, name_file + ".txt"), 'w') as f:
        with redirect_stdout(f):
            model.summary()
    # dump results
    print_str = get_stats_string(accuracy, macro_f1, micro_f1,None,None,None,None)
    str_head, final_ret = get_excel_format(accuracy, macro_f1, micro_f1)
    with open(os.path.join(name_folder, name_file + ".txt"), 'a') as f:
        f.write(print_str)
        f.write('\n')
        # print results in excel format
        f.write(str_head + '\n')
        f.write(final_ret + '\n')
        f.write('\n')
        f.write("train and test time: " + train_test_time + '\n')

    # assing file name
    config['model_file_name'] = 'NONE'

    print("Model saved...")
    return name_file


def copy_files(source_folder_path, dst_folder, override=True):
    # source folder with the current log
    source_folder_path = os.path.abspath(source_folder_path)

    # name of folder with logs
    log_file_dir = os.path.basename(os.path.normpath(source_folder_path))

    # list of files in source folder
    files = os.listdir(source_folder_path)

    dst_folder = os.path.join(dst_folder, log_file_dir)

    if os.path.exists(dst_folder) is not True:
        os.mkdir(dst_folder)

    for file in files:
        tmp_src = os.path.join(source_folder_path,file)
        tmp_dst = os.path.join(dst_folder,file)
        tmp_dst = os.path.splitext(tmp_dst)[0]
        tmp_dst = tmp_dst + '.evt'

        # if file exists but should not be override, continue
        if os.path.exists(tmp_dst) and override is False:
            continue

        # if file exists and should be override remove it
        elif os.path.exists(tmp_dst) and override is True:
            os.remove(tmp_dst)
        # pokud doje k chybe je to tim ze delka cesty ve windows muze byt maximalne 259/269 znaku
        # gpedit.msc
        # https://www.tenforums.com/tutorials/79976-open-local-group-policy-editor-windows-10-a.html
        # https://superuser.com/questions/1119883/windows-10-enable-ntfs-long-paths-policy-option-missing
        # zapnout long file mode
        shutil.copy(tmp_src,tmp_dst)


def save_data_pickle(data, path):
    # save data to a file
    with open(path, 'wb') as fp:
        pickle.dump(data, fp, protocol=4)
    # print("Data successfully saved")


# loading binary data from the given path
def load_data_pickle(path):
    try:
        with open(path, 'rb') as fp:
            data = pickle.load(fp)
    except:
        print('Error reading data')
        print('One more attemp')
        with open(path, 'rb') as fp:
            data = pickle.load(fp)

    return data

