# class based on https://github.com/cbaziotis/datastories-semeval2017-task6
import os

import numpy as np
from nltk.tokenize import WhitespaceTokenizer
# from nltk.tokenize.regexp import WhitespaceTokenizer

from nn.nn_utils import save_data_pickle, load_data_pickle

class EmbeddingsVectorizer:
    def __init__(self,
                 word_map,
                 we_matrix,
                 language,
                 max_length,
                 normalized_tokens=None,
                 tokenizer=None,
                 unk_policy="random"):

        """

        :param word_map:
        :param we_matrix:
        :param language:
        :param max_length:
        :param normalized_tokens: list of tokens <money>, that will be unmasked i.e. "<user>" => "user"
        :param unk_policy: "random","zero"
        """


        # indices to we_matrix
        self.word_map = word_map

        # we matrix
        self.we_matrix = we_matrix

        # todo if is set language use word_tokenize and set language
        self.language = language
        if tokenizer is None:
            self.tokenizer = WhitespaceTokenizer()
        else:
            self.tokenizer = tokenizer

        # max sequence length
        self.max_seq_length = max_length

        self.normalized_tokens = normalized_tokens

        self.unk_policy = unk_policy

        self.OOV = 0

        self.total_words = 0

        self.OOV_words = dict()



    def tokenize_to_list(self,text):
        # tokenizace
        # todo mozna odstranit interpunkci, odchytit OOV slova, mozna sentences
        # words = [x for x in word_tokenize(text, language=self.language, preserve_line=True) if len(x) >= 1]
        words = [x for x in self.tokenizer.tokenize(text) if len(x) >= 1]

        return words

    # prevede text na sekvenci čísel (odpovídajících slovům)
    def text_to_sequence(self,word_list, add_tokens=True):
        max_len = self.max_seq_length


        words = np.zeros(max_len).astype(int)
        # trim tokens after max length
        sequence = word_list[:max_len]


        if add_tokens:
            index = self.word_map.get('<s>', -1)
            if index != -1:
                words[0] = index
                start_token_added = True
            else:
                # bcs we didnt added word
                start_token_added = False



        for i, token in enumerate(sequence):
            index = i
            if add_tokens and start_token_added:
                index = i + 1

            if index >= max_len:
                # todo mozna dodelat protoze pokud je ten text delsi nez max len tak se tam neprida ukoncovaci znackaa
                break

            self.total_words += 1

            # unmask tokens
            if self.normalized_tokens is not None:
                token = self.unmask_token(token)

            if token in self.word_map:
                words[index] = self.word_map[token]
            else:
                tmp = self.remove_inter(token)
                lower = tmp.lower()
                if tmp in self.word_map:
                    # if start with comma or ends with dot
                    words[index] = self.word_map[tmp]
                elif lower in self.word_map:
                    words[index] = self.word_map[lower]

                elif self.unk_policy == 'random':
                    words[index] = self.word_map["<unk>"]
                    # print(token)
                    self.OOV_words[token] = self.OOV_words.get(token, 0) + 1
                    self.OOV += 1
                elif self.unk_policy == 'zero':
                    words[index] = 0
                    # print(token)
                    self.OOV_words[token] = self.OOV_words.get(token, 0) + 1
                    self.OOV += 1

        if add_tokens:
            index = min(index+1,max_len-1)
            words[index] = self.word_map.get('</s>', 0)

        return words

    def unmask_token(self,token):
        for tmp in self.normalized_tokens:
            if tmp == token:
                token = token[1:-1]
                break
        return token

    def remove_inter(self,token):
        if len(token) > 1:
            if token.endswith('.'):
                token = token[:-1]

            if token.startswith('‚'):
                token = token[1:]
        return token

    def print_OOV(self):
        if self.total_words != 0:
            print('OOV words:  ', self.OOV)
            print('Total words:', self.total_words )
            print('Ratio:      ', (self.OOV / self.total_words))

    def print_OOV_words(self):
        for key, value in self.OOV_words.items():
            print(key,' - ',value)



    def vectorize(self,x_texts,y_labels,cache_file_x=None,cache_file_y=None):

        if (cache_file_x is not None) and (cache_file_y is not None):
            if os.path.exists(cache_file_x) and os.path.exists(cache_file_y):
                x_vectors = load_data_pickle(cache_file_x)
                y_vectors = load_data_pickle(cache_file_y)
                return x_vectors, y_vectors

        if len(x_texts) != len(y_labels):
            raise ValueError("x and y must have the same length")

        x_vectors = np.zeros(shape=(len(x_texts),self.max_seq_length))
        # y_vectors = np.zeros(shape=(len(x_texts), len(classes)))
        y_vectors = np.zeros(shape=(len(x_texts), 50)) # we should not need to know it

        for i, (text, label) in enumerate(zip(x_texts,y_labels)):
            # label_vector = create_labels_one_hot_vector(label,classes_indices)
            word_list = self.tokenize_to_list(text)
            x_vector = self.text_to_sequence(word_list)

            # y_vectors[i] = label_vector
            y_vectors[i] = 0
            x_vectors[i] = x_vector

        if (cache_file_x is not None) and (cache_file_y is not None):
            save_data_pickle(x_vectors,cache_file_x)
            save_data_pickle(y_vectors, cache_file_y)

        return x_vectors, y_vectors


