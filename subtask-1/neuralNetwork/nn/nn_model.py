from keras import Model
import keras
# comment this to make results "random" init of network will be random
from keras.callbacks import EarlyStopping
from numpy.random import seed

from utils import create_eval_file

seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
# without this I was not able to run it from command line
import sys
from time import time

import keras
from keras import Model
from keras.layers import *
from keras.layers import Embedding, TimeDistributed, LSTM, Dropout, Bidirectional, Dense, concatenate
from keras.layers.core import *
from keras.optimizers import Adam
from keras.regularizers import l2
from kutilities.layers import Attention
from sklearn.model_selection import train_test_split

from embeddings.EmbeddingsMatrixManager import EmbeddingsMatrixManager
from embeddings.EmbeddingsVectorizer import EmbeddingsVectorizer
from eval import label_probabilities, de_vectorize
from nn.lang_utils import get_lang_configs
from nn.nn_utils import generate_file_name

print(sys.path)
sys.path.append('../')

from config import *
from nn.sequence_utils import extract_seq_words_v2, extract_seq_entire_sentence, \
    extract_n_gram_seq

# dialects
# Dialects = 'ALE ALG ALX AMM ASW BAG BAS BEI BEN CAI DAM DOH FES JED JER KHA MOS MSA MUS RAB RIY SAL SAN SFX TRI TUN'.split()

# create mapping for dialects
dialect2label = {}
label2dialect = {}


def build_and_run(x_train, y_train, x_test, y_test, vectorizers_dict, config, x_train_lang_features_df, x_test_lang_features_df,
                  x_train_word_lang_features,x_test_word_lang_features):
    # x_train, _, y_train, _, x_train_lang_features_df, _ = train_test_split(x_train, y_train, x_train_lang_features_df, x_train_word_lang_features,x_test_word_lang_features,test_size=0.95, random_state=2018)

    max_sentence_len = config['max_sentence_len']
    batch_size = config['batch_size']
    epochs = config['epoch_count']


    input_X_train_emb, input_X_test_emb, we_matrix = extract_word_embeddings(x_train, y_train, x_test, y_test, config)
    input_X_train, input_X_test = extract_sequences(x_train, x_test, vectorizers_dict, max_sentence_len)

    model_getter = build_attention_RNN
    model_getter_args = (vectorizers_dict, we_matrix)

    callback_list, tensorboard_log_dir = init_callbacks(config,config['use_early_stopping'])

    t0 = time()

    X_train = create_data_for_input(input_X_train_emb, input_X_train, x_train_lang_features_df,x_train_word_lang_features, config)
    X_test = create_data_for_input(input_X_test_emb, input_X_test, x_test_lang_features_df,x_test_word_lang_features, config)



    # build model
    model = model_getter(*model_getter_args,**config)
    model = start_training(model, X_train, y_train, X_test, y_test,
                           batch_size, epochs,callback_list)


    y_pred = model.predict(X_test)
    y_pred = label_probabilities(y_pred)
    y_pred = de_vectorize(y_pred)
    y_test = de_vectorize(y_test)

    # time measure
    train_test_time = time() - t0
    train_test_time = "{0:.2f}s".format(train_test_time)

    # return predictions and test
    return y_pred, y_test, model, tensorboard_log_dir, train_test_time, callback_list


def create_data_for_input(input_X_emb, input_X_char, input_X_lang_features, input_X_lang_word_features, config):
    inputs = []

    use_word_emb = config['use_word_emb']
    use_ngram = config['use_ngram']
    use_lang_features = config['use_lang_features']
    use_word_features = config['use_word_features']

    if use_word_emb is True:
        inputs.append(input_X_emb)

    if use_ngram is True:
        inputs.extend(input_X_char)

    if use_lang_features is True:
        tmp = input_X_lang_features.values
        if use_word_features is True:
            tmp_words = input_X_lang_word_features.values
            tmp = np.concatenate((tmp, tmp_words), axis=1)

        inputs.append(tmp)


    return inputs


def extract_word_embeddings(x_train, y_train, x_test, y_test, config):
    train_combination = config['train_data']
    test_combination = config['test_data']
    language = 'ar'

    embeddings_lang_dir, cached_embeddings_path, \
    cached_we_matrix_path, cached_wordmap_path, \
    cached_x_vectors_path, cached_y_vectors_path, \
    cached_x_vectors_test_path, cached_y_vectors_test_path, \
    cached_part_prefix = get_lang_configs(config['embeddings_file'],
                                          train_combination, test_combination, language=language)

    embeddings_dimension = config['embeddings_dimension']
    max_sequence_length = config['max_sequence_length']
    embeddings_path = os.path.join(embeddings_lang_dir, config['embeddings_file'])
    use_gzip = config['use_gzip']
    # get name by embeddings folder name
    embeddings_name = config['embeddings_file']

    we_matrix, wordmap = EmbeddingsMatrixManager(embeddings_filename=embeddings_path,
                                                 cached_embeddings_filename=cached_embeddings_path,
                                                 we_embedding_matrix_filename=cached_we_matrix_path,
                                                 wordmap_filename=cached_wordmap_path,
                                                 dimension=embeddings_dimension,
                                                 use_gzip=use_gzip).get_we_matrix()

    vectorizer = EmbeddingsVectorizer(word_map=wordmap,
                                      we_matrix=we_matrix,
                                      language=language,
                                      max_length=max_sequence_length)

    x_vectors_train, _ = vectorizer.vectorize(x_train, y_train
                                                            ,cache_file_x=cached_x_vectors_path
                                                            ,cache_file_y=cached_y_vectors_path)

    x_vectors_test, _ = vectorizer.vectorize(x_test, y_test,
                                                          cache_file_x=cached_x_vectors_test_path,
                                                          cache_file_y=cached_y_vectors_test_path)
    vectorizer.print_OOV()

    return x_vectors_train, x_vectors_test, we_matrix


def init_callbacks(config, use_early_stop=False):
    embeddings_name = config['embeddings_file']

    batch_size = config['batch_size']

    file_name = generate_file_name(config, embeddings_name)
    tensorboard_log_dir = os.path.join(BASE_PATH, 'logs')
    tensorboard_log_dir = os.path.join(tensorboard_log_dir, file_name)
    os.makedirs(tensorboard_log_dir)
    tensorBoardCB = keras.callbacks.TensorBoard(log_dir=tensorboard_log_dir, histogram_freq=0, batch_size=batch_size,
                                                write_graph=True, write_grads=True,
                                                write_images=False, embeddings_freq=0, embeddings_layer_names=None,
                                                embeddings_metadata=None)

    callbacks_list = [tensorBoardCB]
    if use_early_stop:
        early_stop = EarlyStopping(monitor='val_loss', patience=0, mode='min')
        callbacks_list.append(early_stop)

    return callbacks_list, tensorboard_log_dir

def start_training(model, x_train, y_train, x_validate, y_validate, batch_size,
                epoch_count, callback_list):
    print("Training model...")
    model.fit(x_train,
                y_train,
                batch_size=batch_size,
                epochs=epoch_count,
                validation_data=(x_validate, y_validate),
                shuffle=True,
                callbacks=callback_list,
                verbose=1)
    return model


def extract_sequences(x_train, x_test, vectorizers_dict, max_sentence_len):

    # ngram settings
    extracted_X_dict_train = extract_n_gram_seq(x_train, vectorizers_dict, max_sentence_len=max_sentence_len)
    extracted_X_dict_test = extract_n_gram_seq(x_test, vectorizers_dict, max_sentence_len=max_sentence_len)

    inputs_ngram_X_train = []
    inputs_ngram_X_test = []

    # get extracted sequences
    for ngram_order, vectorizer in vectorizers_dict.items():
        inputs_ngram_X_train.append(extracted_X_dict_train.get(ngram_order))
        inputs_ngram_X_test.append(extracted_X_dict_test.get(ngram_order))

    # input_X_train = []
    input_X_train = []
    input_X_test = []

    input_X_train.extend(inputs_ngram_X_train)
    input_X_test.extend(inputs_ngram_X_test)

    return input_X_train, input_X_test


def build_attention_RNN(vectorizers_dict, we_matrix, unit=LSTM, **kwargs):
    print("Model configuration:")
    for i, v in kwargs.items():
        print(i, ": ", v)

    # general settings
    use_ngram = kwargs.get("use_ngram", True)
    use_word_emb = kwargs.get("use_word_emb", True)
    bi = kwargs.get("bidirectional", False)
    final_layer = kwargs.get("final_layer", False)  # if add final_layer
    attention = kwargs.get("attention", False)  # if use attention
    decay_rate = kwargs.get("decay_rate", 0)  # decay rate for the optimizer
    use_masking = kwargs.get("use_masking", True)
    noise = kwargs.get("noise", 0.)  # word embeddings noise

    # general dropout
    dropout_rnn = kwargs.get("dropout_rnn", 0)  # dropout after each RNN layer
    dropout_rnn_recurrent = kwargs.get("dropout_rnn_recurrent", 0)  # dropout for recurrent connections
    dropout_attention = kwargs.get("dropout_attention", 0)  # dropout after attention layer
    dropout_final = kwargs.get("dropout_final", 0)  # dropout after final layer

    rnn_loss_l2 = kwargs.get("rnn_loss_l2", 0.)  # recurent l2 regularization
    loss_l2 = kwargs.get("loss_l2", 0.)  # final layer l2 regularizer

    # variables for characters
    trainable_chars = kwargs.get("trainable_chars", True)  # if the we layer is trainable
    dropout_chars = kwargs.get("dropout_chars", 0)  # dropout after character embeddings
    ngram_rnn_layers = kwargs.get("ngram_rnn_layers", 2)  # number of ngram RNN layers
    ngram_cells = kwargs.get("ngram_cells", 10)  # number of unit in ngram RNN cells
    max_sentence_len = kwargs.get("max_sentence_len", 200)  # max sentence len in chars
    char_emb_output_dim = kwargs.get("char_emb_output_dim", 150)  # output dimension of character emb layer

    # variables for embedding
    trainable_we = kwargs.get("trainable_we", False)  # if the we layer is trainable
    dropout_words = kwargs.get("dropout_words", 0)  # dropout after embeddings
    rnn_layers = kwargs.get("rnn_layers", 2)  # number of RNN layers
    cells = kwargs.get("cells", 64)  # number of unit in RNN cells
    max_sequence_length = kwargs.get("max_sequence_length", 64)  # max sequence length


    # variable for lang features
    use_lang_features = kwargs.get("use_lang_features", False)
    num_lang_dense_layers = kwargs.get("num_lang_dense_layers", 2)
    dense_lang_layers_cells = kwargs.get("dense_lang_layers_cells",400)
    dense_lang_dropout = kwargs.get("dense_lang_dropout",0)
    lang_features_len = kwargs.get("lang_features_len",26)

    out_len = kwargs.get('out_len',26)
    lr = kwargs.get("lr", 0.0005)
    optimizer_name = kwargs.get("optimizer", 'Adam')  # default optimizer is Adam

    # get object of optimizer
    optimizer = get_optimizer(optimizer_name, lr, decay_rate)

    # build ngram part of nn
    encoding_out = []
    inputs = []
    concat_char = None
    emb_word = None
    lang_nn = None

    if use_word_emb is True:
        emb_in = Input(shape=(max_sequence_length,), name='emb-input')
        inputs.append(emb_in)

        emb_word = (get_embeddings_layer(we_matrix, max_sequence_length, trainable_we=trainable_we, use_masking=use_masking))(emb_in)

        if noise > 0:
            emb_word = GaussianNoise(noise)(emb_word)

        if dropout_words > 0:
            emb_word = Dropout(dropout_chars)(emb_word)

        # RNN layers
        for i in range(rnn_layers):
            rs = (rnn_layers > 1 and i < rnn_layers - 1) or attention
            emb_word = (get_rnn_layer(unit, cells, bi,
                                      return_sequences=rs,
                                      recurent_dropout=dropout_rnn_recurrent,
                                      l2_reg=rnn_loss_l2))(emb_word)
            if dropout_rnn > 0:
                emb_word = (Dropout(dropout_rnn))(emb_word)

        # Attention after RNN
        if attention is True:
            emb_word = Attention()(emb_word)
            if dropout_attention > 0:
                emb_word = Dropout(dropout_attention)(emb_word)


    if use_ngram is True:
        for ngram_order, vectorizer in vectorizers_dict.items():
            vocab = vectorizer.vocabulary_
            vocab_size = len(vocab) + 3
            print('Vocab size:' + str(vocab_size))

            char_in = Input(shape=(max_sentence_len,), name='char-input-' + str(ngram_order))

            emb_char = Embedding(input_dim=vocab_size, output_dim=char_emb_output_dim,
                                 input_length=max_sentence_len, mask_zero=use_masking,
                                 trainable=trainable_chars)(char_in)
            if noise > 0:
                emb_char = GaussianNoise(noise)(emb_char)

            if dropout_chars > 0:
                emb_char = Dropout(dropout_chars)(emb_char)

            # RNN layers
            for i in range(ngram_rnn_layers):
                rs = (ngram_rnn_layers > 1 and i < ngram_rnn_layers - 1) or attention
                emb_char = (get_rnn_layer(unit, ngram_cells, bi,
                                          return_sequences=rs,
                                          recurent_dropout=dropout_rnn_recurrent,
                                          l2_reg=rnn_loss_l2))(emb_char)
                if dropout_rnn > 0:
                    emb_char = (Dropout(dropout_rnn))(emb_char)

            # Attention after RNN
            if attention is True:
                emb_char = Attention()(emb_char)
                if dropout_attention > 0:
                    emb_char = Dropout(dropout_attention)(emb_char)

            encoding_out.append(emb_char)
            inputs.append(char_in)

        if len(encoding_out) < 2:
            concat_char = encoding_out[0]
        else:
            concat_char = concatenate(encoding_out)

    if use_lang_features is True:
        lang_in = Input(shape=(lang_features_len,), name='lang-input')
        inputs.append(lang_in)

        lang_nn = lang_in

        for i in range(num_lang_dense_layers):
            lang_nn = (Dense(dense_lang_layers_cells,activation='relu'))(lang_nn)

            if dense_lang_dropout > 0:
                lang_nn = Dropout(dense_lang_dropout)(lang_nn)




    layers = None
    if (use_word_emb is True) and (use_ngram is True) and (use_lang_features is True):
        layers = concatenate([emb_word, concat_char, lang_nn])
    elif (use_word_emb is True) and (use_ngram is True):
        layers = concatenate([emb_word, concat_char])
    elif (use_word_emb is True) and (use_lang_features is True):
        layers = concatenate([emb_word, lang_nn])
    elif (use_ngram is True) and (use_lang_features is True):
        layers = concatenate([concat_char, lang_nn])
    elif use_word_emb is True:
        layers = emb_word
    elif use_ngram:
        layers = concat_char
    elif use_lang_features:
        layers = lang_nn
    else:
        raise Exception('Unknown configuration')


    if final_layer is True:
        layers = Dense(400, activation='relu')(layers)
        if dropout_final > 0:
             layers = Dropout(dropout_final)(layers)

    out = Dense(out_len, activation='softmax', activity_regularizer=l2(loss_l2))(layers)

    model = Model(inputs=inputs, outputs=[out])
    model.summary()
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["acc"])

    return model


def get_embeddings_layer(we_matrix, max_len, trainable_we=False, use_masking=True):
    # size of word embeddings, number of words for which embedding vector exist
    vocab_size = we_matrix.shape[0]
    we_dimension = we_matrix.shape[1]

    emb_layer = Embedding(
        input_dim=vocab_size,
        weights=[we_matrix],
        output_dim=we_dimension,
        input_length=max_len,
        trainable=trainable_we,
        mask_zero=use_masking
    )

    return emb_layer

def get_rnn_layer(unit, cells=64, bi=False, return_sequences=True, recurent_dropout=0., l2_reg=0):
    rnn = unit(cells, return_sequences=return_sequences,
               #recurrent_dropout=recurent_dropout,
               kernel_regularizer=l2(l2_reg))
    if bi is True:
        return Bidirectional(rnn)
    else:
        return rnn


def get_optimizer(optimizer_name, lr, decay_rate):
    if optimizer_name == 'Adam':
        optimizer = keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=decay_rate,
                                          amsgrad=False)
    elif optimizer_name == 'RMSprop':
        optimizer = keras.optimizers.RMSprop(lr=lr, rho=0.9, epsilon=None, decay=decay_rate)

    elif optimizer_name == 'Adagrad':
        optimizer = keras.optimizers.Adagrad(lr=lr, epsilon=None, decay=decay_rate)

    elif optimizer_name == 'Adadelta':
        optimizer = keras.optimizers.Adadelta(lr=lr, rho=0.95, epsilon=None, decay=decay_rate)

    elif optimizer_name == 'SGD':
        optimizer = keras.optimizers.SGD(lr=lr, momentum=0.0, decay=decay_rate, nesterov=False)
    else:
        raise Exception('Not valid optimizer:' + optimizer_name)

    return optimizer


