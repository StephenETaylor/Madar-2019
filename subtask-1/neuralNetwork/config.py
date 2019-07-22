import os

# data_cols_s1 = ['sent', 'label']
data_cols_s1 = ['sent', 'label']
data_test_cols_s1 = ['sent']
feature_cols_s1 = []
for i in range(26):
    feature_name = 'f-' + str(i)
    feature_cols_s1.append(feature_name)
    data_cols_s1.append(feature_name)
    data_test_cols_s1.append(feature_name)


text_col = 'sent'
label_col = 'label'

DATE_SAVING_FORMAT = '%Y/%m/%d %H:%M:%S'

u_flag_col = 'unique'
u_label_col = 'unique_label'

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
TMP_DIR = os.path.join(BASE_PATH, 'tmp')
TRAIN_MOD_DIR = os.path.join(BASE_PATH, "trained_models")

RESULTS_FOLDER = os.path.join(BASE_PATH,'results')

M_DIR_S1 = os.path.join(BASE_PATH,'MADAR-Shared-Task-Subtask-1')

# class fo loading dataset from config file
class DATASET(object):
    path = {
        'TRAIN_DATA_S1' : os.path.join(M_DIR_S1,'MADAR-Corpus-26-train.lm26c'),
        # 'TRAIN_DATA_S1' : os.path.join(M_DIR_S1,'MERGED-26-6-train.lm26c'),
        'DEV_DATA_S1' : os.path.join(M_DIR_S1,'MADAR-Corpus-26-dev.lm26c'),
        'GOLD_PRED_FILE' : os.path.join(RESULTS_FOLDER,'dev-26.GOLD'),
        'TEST_DATA_S1' : os.path.join(M_DIR_S1, 'MADAR-Corpus-26-test.lm26c'),
        'TRAIN_DATA_S1_WORDS': os.path.join(M_DIR_S1, 'MADAR-Corpus-26-train.lm26w'),
        'DEV_DATA_S1_WORDS': os.path.join(M_DIR_S1, 'MADAR-Corpus-26-dev.lm26w')
    }


EMB_DIR = os.path.join(BASE_PATH, "emb")
TMP_SPLIT_DIR = os.path.join(TMP_DIR, "split")

class LANG_config(object):
    def __init__(self, embeddings_file, train_combination, test_combination, lang='en'):
        self.lang = lang

        embeddings_file = embeddings_file.replace('/','-')
        self.embeddings_lang_dir = os.path.join(EMB_DIR, lang)
        self.cached_embeddings_path = os.path.join(TMP_DIR, embeddings_file + '-' + lang + '-emb.bin')
        self.cached_we_matrix_path = os.path.join(TMP_DIR, embeddings_file + '-' + lang + '-we_matrix.bin')
        self.cached_wordmap_path = os.path.join(TMP_DIR, embeddings_file + '-' + lang + '-wordmap.bin')
        self.cached_x_vectors_path = os.path.join(TMP_DIR, embeddings_file + '-' + train_combination + '-' + lang + '-cached_x.bin')
        self.cached_y_vectors_path = os.path.join(TMP_DIR, embeddings_file + '-' + train_combination + '-' + lang + '-cached_y.bin')
        self.cached_x_vectors_test_path = os.path.join(TMP_DIR, embeddings_file + '-' + test_combination + '-' + lang + '-cached_test_x.bin')
        self.cached_y_vectors_test_path = os.path.join(TMP_DIR, embeddings_file + '-' + test_combination + '-' + lang + '-cached_text_y.bin')
        self.cached_part_prefix = os.path.join(TMP_SPLIT_DIR, embeddings_file + '-' + train_combination + '-' + lang +  '_')


    @property
    def cached_part_prefix(self):
        return self._cached_part_prefix

    @cached_part_prefix.setter
    def cached_part_prefix(self, value):
        self._cached_part_prefix = value

    @property
    def lang(self):
        return self._lang

    @lang.setter
    def lang(self, value):
        self._lang = value

    @property
    def embeddings_lang_dir(self):
        return self._embeddings_lang_dir

    @embeddings_lang_dir.setter
    def embeddings_lang_dir(self, value):
        self._embeddings_lang_dir = value

    @property
    def cached_embeddings_path(self):
        return self._cached_embeddings_path

    @cached_embeddings_path.setter
    def cached_embeddings_path(self, value):
        self._cached_embeddings_path = value

    @property
    def cached_we_matrix_path(self):
        return self._cached_we_matrix_path

    @cached_we_matrix_path.setter
    def cached_we_matrix_path(self, value):
        self._cached_we_matrix_path = value

    @property
    def cached_wordmap_path(self):
        return self._cached_wordmap_path

    @property
    def cached_x_vectors_path(self):
        return self._cached_x_vectors_path

    @property
    def cached_y_vectors_path(self):
        return self._cached_y_vectors_path

    @property
    def cached_x_vectors_test_path(self):
        return self._cached_x_vectors_test_path

    @property
    def cached_y_vectors_test_path(self):
        return self._cached_y_vectors_test_path

    @cached_x_vectors_path.setter
    def cached_x_vectors_path(self, value):
        self._cached_x_vectors_path = value

    @cached_y_vectors_path.setter
    def cached_y_vectors_path(self, value):
        self._cached_y_vectors_path = value

    @cached_x_vectors_test_path.setter
    def cached_x_vectors_test_path(self, value):
        self._cached_x_vectors_test_path = value

    @cached_y_vectors_test_path.setter
    def cached_y_vectors_test_path(self, value):
        self._cached_y_vectors_test_path = value

    @cached_wordmap_path.setter
    def cached_wordmap_path(self, value):
        self._cached_wordmap_path = value