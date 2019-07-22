from config import *


def get_lang_configs(embeddings_file, train_combination, test_combination, language):
    dataset = LANG_config(embeddings_file, train_combination, test_combination, lang=language)
    embeddings_lang_dir = dataset.embeddings_lang_dir
    cached_embeddings_path = dataset.cached_embeddings_path
    cached_we_matrix_path = dataset.cached_we_matrix_path
    cached_wordmap_path = dataset.cached_wordmap_path
    cached_x_vectors_path = dataset.cached_x_vectors_path
    cached_y_vectors_path = dataset.cached_y_vectors_path
    cached_x_vectors_test_path = dataset.cached_x_vectors_test_path
    cached_y_vectors_test_path = dataset.cached_y_vectors_test_path
    cached_part_prefix = dataset.cached_part_prefix

    return embeddings_lang_dir, cached_embeddings_path,\
           cached_we_matrix_path,cached_wordmap_path,\
           cached_x_vectors_path, cached_y_vectors_path,\
           cached_x_vectors_test_path,cached_y_vectors_test_path,\
           cached_part_prefix

