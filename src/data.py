import os
import pickle

import math

import gc

import collections
import numpy as np
import time
from gensim.models import Word2Vec
from gensim.models.fasttext import FastText
from keras.layers import Embedding
from keras.models import Sequential
from keras.utils import to_categorical
import re
import random

from liner2 import Liner2, start_jvm
from maltparser import MaltParser
from utils import sliding_window, balanced_split, split


class Data(object):

    def __init__(self, config, input_files_index, window_size,
                 irrelevant_class, single_class, token_sequences_file_name,
                 file_to_save_token_sequences, cv=False, input_format="batch:ccl"):
        self.document_limit = None
        self.config = config
        self.cv = cv
        self.file_to_save_token_sequences = file_to_save_token_sequences
        self.input_files_index = input_files_index
        self.input_format = input_format
        self.irrelevant_class = irrelevant_class
        self.single_class = single_class
        self.token_sequences_file_name = token_sequences_file_name
        self.window_size = window_size
        self.max_sequence_length = self.window_size * 2 + 1

        start_jvm([config['maltparser']['jar'], config['liner']['jar']], [config['liner']['lib'], config['maltparser']['lib']])
        self.dependency_parser = MaltParser(config['maltparser']['model'])

        self.liner = Liner2(config['liner']['config'])
        self.annotation_types = self.liner.options.getTypes()

        self.w2v_embeddings = {}
        self.ft_embeddings = {}
        self.indexed_embedding_dicts = {}
        self.position_embedding_dict ={}
        self.input_dim = 0
        self.input_dims = [0]
        self.num_classes = 0

        self.max_sentence_len = 0
        self.indexed_features = {}
        self.labels_index = {}

        self.sliding_window_limited_to_sentence = False

        self.features_to_index = []
        if config['indexed_embeddings'] and len(config['indexed_embeddings']):
            self.features_to_index = [e for e in config['indexed_embeddings']]

        self.annotation_to_label_mapper = None
        if single_class:
            self.annotation_to_label_mapper = lambda (a): "event"


    def load(self, indexed_features_file=None):
        print('load data')
        self.data = []

        seq_dict = self.load_token_sequences(indexed_features_file)
        sequences = seq_dict['sequences']


        max_sentence_len = seq_dict['max_sentence_len']
        self.num_classes = len(self.labels_index)
        print('Found %s classes.' %  self.num_classes)
        print('max_sentence_len %s.' % max_sentence_len)

        self.load_embeddings()
        print('after load_embeddings')

        self.sequences = sequences

        self.input_dims = [self.input_dim]

        gc.collect()
        print('data loaded and mapped')

    def get_label(self, seq):
        return self.get_label_from_idx(seq[0])

    def get_label_from_idx(self, label_idx):
        return to_categorical(label_idx, self.num_classes)[0]

    def get_vector_sequence(self, seq):
        return [self.map_token(t, self.input_dim, i, True) for i, t in enumerate(seq[1])]

    def load_token_sequences(self, indexed_features_file):
        if not self.token_sequences_file_name and self.input_files_index:

            self.indexed_features = None
            if indexed_features_file:
                self.load_indexed_features(indexed_features_file)

            seq_dict = self.get_token_sequences(self.input_files_index, cv=self.cv, document_limit=self.document_limit)

            if self.file_to_save_token_sequences:
                print("Saving sequences to file")

                with open(self.file_to_save_token_sequences, 'wb') as f:
                    pickle.dump(seq_dict, f)
        else:
            print("Loading sequences from file")
            with open(self.token_sequences_file_name, "rb") as fp:
                seq_dict = pickle.load(fp)
            self.labels_index = seq_dict['labels_index']
            self.indexed_features = seq_dict['indexed_features']
            self.sequences = seq_dict['sequences']
            self.max_sentence_len = seq_dict['max_sentence_len']

        return seq_dict

    def map_token(self, t, input_d, position=None, add_position_feat=False):
        if not t:
            return np.zeros(input_d)

        vectors = []
        for e_name, e_conf in self.config['w2v_embeddings'].items():
            if not e_conf['enabled']:
                continue

            for attr in e_conf['attributes']:

                if isinstance(attr, list):
                    value = '.'.join([t[a] if t[a] else '' for a in attr])
                else:
                    value = t[attr]

                vector = self.get_w2v_vector(value, self.w2v_embeddings[e_name])
                vectors.append(vector)

        for e_name, e_conf in self.config['ft_embeddings'].items():
            if not e_conf['enabled']:
                continue

            for attr in e_conf['attributes']:

                if isinstance(attr, list):
                    value = '.'.join([t[a] if t[a] else '' for a in attr])
                else:
                    value = t[attr]

                vector = self.get_w2v_vector(value, self.ft_embeddings[e_name])
                vectors.append(vector)

        for attr in self.config['indexed_embeddings']:
            dim = self.config['indexed_embeddings'][attr]

            attr_tmp = attr
            if attr in ['dependency_path_from', 'dependency_path_to']:
                attr_tmp = 'dependency_path_short'

            try:
                vectors.append(self.indexed_embedding_dicts[attr_tmp][t[attr_tmp + '_index']])
            except KeyError:
                vectors.append(np.zeros(dim))

        if self.config['distance_feature']:
            vectors.append(np.array([math.log(t['distance_from']+1)]))
            vectors.append(np.array([math.log(t['distance_to']+1)]))

        if self.config['sentence_distance_feature']:
            vectors.append(np.array([math.log(t['sentence_distance_from']+1)]))
            vectors.append(np.array([math.log(t['sentence_distance_to']+1)]))

        # if add_position_feat:
        #     vectors.append(self.position_embedding_dict[position])

        return np.concatenate(vectors)


    def split_data(self, split_ratio):

        d1 = []
        d2 = []

        d1_target_len = split_ratio * len(self.data[0])

        for doc_seqs in self.data[0][:d1_target_len]:
            d1.extend(doc_seqs)
        for doc_seqs in self.data[0][d1_target_len:]:
            d2.extend(doc_seqs)

        return np.array(d1), np.array(d2)

    def get_training_data(self, validation_split):
        print('get_training_data')

        x_train = []
        y_train = []

        split_ratio = 1 - validation_split

        d1_target_len = int(split_ratio * len(self.sequences))
        print('train_target_len', d1_target_len)



        for doc_seqs, not_matched in self.sequences[:d1_target_len]:
            for seq in doc_seqs:
                x_train.append(self.get_vector_sequence(seq))
                y_train.append(self.get_label(seq))

        # for doc_labels in self.labels[:d1_target_len]:
        #     y_train.extend(doc_labels)

        x_val = []
        y_val = []

        na_class = to_categorical(self.labels_index[self.irrelevant_class], self.num_classes)[0]
        for d_idx in range(d1_target_len, len(self.sequences)):
            doc_seqs, not_matched = self.sequences[d_idx]
            # doc_labels = self.labels[d_idx]

            for s_idx in range(len(doc_seqs)):
                seq = doc_seqs[s_idx]
                label = self.get_label(seq)

                if np.array_equal(label, na_class) and random.random() > self.config['train_candidate_ratio']:
                    continue

                x_val.append(self.get_vector_sequence(seq))
                y_val.append(label)


        x_train = np.array(x_train)
        gc.collect()
        x_val = np.array(x_val)
        gc.collect()
        y_train = np.array(y_train)
        gc.collect()
        y_val = np.array(y_val)

        num_validation_samples = x_val.shape[0]
        num_samples = x_train.shape[0] + num_validation_samples


        print('x_train shape:', x_train.shape)
        print('x_val shape:', x_val.shape)

        print('y_train shape:', y_train.shape)
        print('y_val shape:', y_val.shape)

        return {
            'num_samples': num_samples,
            'num_validation_samples': num_validation_samples,
            'x_train': x_train,
            'x_val': x_val,
            'y_train': y_train,
            'y_val': y_val,
        }

    def get_validation_test_data(self, validation_split):

        all_not_marched = {}

        split_ratio = 1 - validation_split
        d1_target_len = int(split_ratio * len(self.sequences))

        x_val_test = []
        y_val_test = []

        labels_index_inverted = self.get_labels_index_inverted()

        for doc_seqs, not_matched in self.sequences[d1_target_len:]:

            for nm in not_matched:
                try:
                    nm_count = all_not_marched[nm]
                except KeyError:
                    nm_count = 0

                all_not_marched[nm] = nm_count + not_matched[nm]

                # x_val_test.extend(doc_seqs)
            for seq in doc_seqs:

                is_discarded = seq[2]
                if is_discarded:
                    type = labels_index_inverted[seq[0]]
                    try:
                        nm_count = all_not_marched[type]
                    except KeyError:
                        nm_count = 0

                    all_not_marched[type] = nm_count + 1



                x_val_test.append(self.get_vector_sequence(seq))
                y_val_test.append(self.get_label(seq))

        # for doc_labels in self.labels[d1_target_len:]:
        #     y_val_test.extend(doc_labels)



        x_val_test = np.array(x_val_test)
        y_val_test = np.array(y_val_test)

        num_validation_samples = x_val_test.shape[0]

        print('x_val_test shape:', x_val_test.shape)
        print(' val_test all not matched:', all_not_marched)

        return {
            'num_validation_samples': num_validation_samples,
            'x_val_test': x_val_test,
            'y_val_test': y_val_test,
            'not_matched': all_not_marched
        }


    def get_cv_folds_training_data(self, validation_split):

        folds = self.load_folds(self.input_files_index)
        self.folds = folds
        n_folds = len(folds)


        self.input_dims = [self.input_dim]

        if len(self.labels_index) <= 1:
            self.init_rel_types_label_indexes()

            self.num_classes = len(self.labels_index)




        for fold_index, test_fold in enumerate(folds):
            test_index = fold_index

            all_train_fold_indices = [f for f in range(n_folds) if f != test_index]

            train_readers = [self.liner.get_batch_reader("\n".join(folds[i]), "", "cclrel") for i in all_train_fold_indices]
            test_reader = self.liner.get_batch_reader("\n".join(test_fold), "", "cclrel")

            gen = self.get_token_feature_generator()

            self.indexed_features = {}
            for f in self.features_to_index:
                self.indexed_features[f] = {}

            gc.collect()


            all_train_doc_number = sum([len(folds[f_i]) for f_i in all_train_fold_indices])

            print("all_train_doc_number", all_train_doc_number)


            split_ratio = 1 - validation_split

            train_target_len = int(split_ratio * all_train_doc_number)
            print('train_target_len', train_target_len)

            train_sequences = []
            val_sequences = []
            self.lock_feature_indexes = False
            doc_idx = -1
            for reader in train_readers:
                while True:
                    doc_idx += 1
                    document = reader.nextDocument()
                    if document is None:
                        break

                    is_train_set = doc_idx < train_target_len
                    self.lock_feature_indexes = not is_train_set

                    doc_seqs = self.process_document(document, gen, use_candidate_ratio=True)

                    if is_train_set:
                        train_sequences.append(doc_seqs)
                    else:
                        val_sequences.append(doc_seqs)

            self.num_classes = len(self.labels_index)
            self.load_embeddings()
            self.input_dims = [self.input_dim]


            print('after load_embeddings')

            x_train = []
            y_train = []

            for doc_seqs, not_matched in train_sequences:
                for seq in doc_seqs:
                    x_train.append(self.get_vector_sequence(seq))
                    y_train.append(self.get_label(seq))

            x_val = []
            y_val = []

            for doc_seqs, not_matched in val_sequences:
                for seq in doc_seqs:
                    x_val.append(self.get_vector_sequence(seq))
                    y_val.append(self.get_label(seq))

            x_train = np.asarray(x_train)
            y_train = np.asarray(y_train)
            x_val = np.asarray(x_val)
            y_val = np.asarray(y_val)

            num_validation_samples = x_val.shape[0]
            num_samples = x_train.shape[0] + num_validation_samples



            print('x_train shape:', x_train.shape)
            print('y_train shape:', y_train.shape)
            print('x_val shape:', x_val.shape)
            print('y_val shape:', y_val.shape)



            yield {
                'num_samples': num_samples,
                'num_validation_samples': num_validation_samples,
                'fold_index': fold_index,
                'x_train': x_train,
                'x_val': x_val,
                'y_train': y_train,
                'y_val': y_val,
                'test_reader': test_reader
        }

    def get_cv_fold_test_data(self, test_reader):

        sequences = []
        gen = self.get_token_feature_generator()
        self.lock_feature_indexes = True
        while True:
            document = test_reader.nextDocument()
            if document is None:
                break

            sequences.append(self.process_document(document, gen, use_candidate_ratio=False))


        all_not_marched = {}

        x_val_test = []
        y_val_test = []

        labels_index_inverted = self.get_labels_index_inverted()

        for doc_seqs, not_matched in sequences:

            for nm in not_matched:
                try:
                    nm_count = all_not_marched[nm]
                except KeyError:
                    nm_count = 0

                all_not_marched[nm] = nm_count + not_matched[nm]

                # x_val_test.extend(doc_seqs)
            for seq in doc_seqs:

                is_discarded = seq[2]
                if is_discarded:
                    type = labels_index_inverted[seq[0]]
                    try:
                        nm_count = all_not_marched[type]
                    except KeyError:
                        nm_count = 0

                    all_not_marched[type] = nm_count + 1



                x_val_test.append(self.get_vector_sequence(seq))
                y_val_test.append(self.get_label(seq))

        # for doc_labels in self.labels[d1_target_len:]:
        #     y_val_test.extend(doc_labels)


        x_val_test = np.array(x_val_test)
        y_val_test = np.array(y_val_test)

        num_validation_samples = x_val_test.shape[0]

        print('x_val_test shape:', x_val_test.shape)
        print(' val_test all not matched:', all_not_marched)

        return {
            'num_validation_samples': num_validation_samples,
            'x_test': x_val_test,
            'y_test': y_val_test,
            'not_matched': all_not_marched
        }

    def get_w2v_vector(self, word, w2v):
        if word and word in w2v:
            return w2v[word]
        else:
            return np.zeros(w2v.vector_size)

    def map_label(self, t, labels_index, irrelevant_class, binary_not_categorical=False):
        num_classes = len(labels_index)

        if num_classes == 2 and binary_not_categorical:
            if t:
                return t['label_index']
            else:
                return labels_index[irrelevant_class]
        if t:
            return to_categorical(t['label_index'], num_classes)[0]
        else:
            return to_categorical(labels_index[irrelevant_class], num_classes)[0]

    def map_label_cv(self, t, labels_index, irrelevant_class, binary_not_categorical=False):
        num_classes = len(labels_index)

        if num_classes == 2 and binary_not_categorical:
            if t:
                return t['label_index']
            else:
                return labels_index[irrelevant_class]
        if t:
            return to_categorical(t['label_index'], num_classes)[0]
        else:
            return to_categorical(labels_index[irrelevant_class], num_classes)[0]


    def load_folds(self, input_file):

        folds = []
        with open(input_file) as f:
            lines = f.readlines()

        root = os.path.dirname(input_file)

        for line in lines:
            line_data = line.strip().split('\t')
            if len(line_data) != 2:
                print("Incorrect line in folds file: " + input_file, line)
                continue
            file_name, fold = line_data
            if not file_name.startswith("/"):
                file_name = root + "/" + file_name
            while len(folds) < int(fold):
                folds.append([])

            folds[int(fold) - 1].append(file_name)

        return folds

    def get_training_set(fold, folds):
        pass

    # get sequences of dicts of token features
    def get_token_sequences(self, input_files_index,
                            document_limit=None, cv=False):

        readers = []
        if cv:
            for index, fold in enumerate(self.load_folds(input_files_index)):
                readers.append(self.liner.get_batch_reader("\n".join(fold), "", "cclrel"))
        else:
            readers.append(self.get_reader(input_files_index))

        gen = self.get_token_feature_generator()


        if not self.indexed_features:
            self.indexed_features = {}
            for f in self.features_to_index:
                self.indexed_features[f] = {}

        # labels_index = {}  # dictionary mapping label name to numeric id
        if not len(self.labels_index):
            self.labels_index[self.irrelevant_class] = 0


        all_sequences = []
        for reader in readers:
            all_sequences.append(self.process_reader(reader, gen, document_limit, cv=cv))

        if len(readers) == 1:
            all_sequences = all_sequences[0]

        return {
            'labels_index': self.labels_index,
            'indexed_features': self.indexed_features,
            'sequences': all_sequences,
            'max_sentence_len': self.max_sentence_len
        }

    def get_reader(self, input_files_index=None):
        if not input_files_index:
            input_files_index = self.input_files_index
        return self.liner.get_reader(input_files_index, self.input_format)

    def get_token_feature_generator(self):
        return self.liner.get_token_feature_generator()

    def process_reader(self, reader, token_features_generator = None, document_limit=None, cv=False):

        if not token_features_generator:
            token_features_generator = self.get_token_feature_generator()

        documents = [] #fixme
        while True:

            document = reader.nextDocument()

            if document is None:
                break
            documents.append(document)

        train_set_len = int((1-self.config['validation_split']) * len(documents))

        sequences = []
        self.lock_feature_indexes = False
        ii = 0
        while True:
            ii += 1
            if document_limit and ii > document_limit:
                break
            if ii > len(documents):
                break
            document = documents[ii-1]


            print('processing document '+str(ii)+'/'+str(len(documents)))
            in_train_set = ii <= train_set_len

            if not in_train_set:
                self.lock_feature_indexes = True
                print("lock_feature_indexes")

            sequences.append(self.process_document(document, token_features_generator, use_candidate_ratio=(cv or (
            in_train_set))))
        return sequences


    def get_annotation_position(self, a):
        return (a.getSentence().getOrd(), a.getHead())


    def get_relation_position(self, r):
        return (self.get_annotation_position(r.getAnnotationFrom()), self.get_annotation_position(r.getAnnotationTo()))

    def compare_positions(self, p1, p2):
        if p1[0] == p2[0]:
            return p1[1] - p2[1]

        return p1[0] - p2[0]

    def get_distance(self, p1, p2):
        l_pos = p1
        r_pos = p2
        if self.compare_positions(p1, p2) > 0:
            l_pos = p2
            r_pos = p1

        if p2[0] == p1[0]:
            rest = r_pos[1] - l_pos[1]
        else:
            rest = r_pos[1]  + self.doc_temp['sentence_lengths'][l_pos[0]] - l_pos[1]

        return rest + sum([self.doc_temp['sentence_lengths'][i] for i in range(l_pos[0]+1 ,r_pos[0])])

    # if beg_token_index=None start from end
    def get_tokens_from_left(self, sentences, num_tokens, current_sentence_index, beg_token_index=None):
        if beg_token_index is not None and beg_token_index < 0:
            current_sentence_index = current_sentence_index - 1

        if current_sentence_index<0:
            return [None for _ in range(num_tokens)]

        sent = sentences.get(current_sentence_index)
        sent_len = sent.getTokens().size()
        if beg_token_index is None:
            beg_token_index = sent_len-1

        tokens = [self.process_token(sent, i) for i in range(max(0, beg_token_index-num_tokens+1), beg_token_index+1)]
        num_tokens_left = num_tokens-len(tokens)
        if num_tokens_left == 0:
            return tokens

        return self.get_tokens_from_left(sentences, num_tokens_left, current_sentence_index-1) + tokens

    def get_tokens_from_right(self, sentences, num_tokens, current_sentence_index, beg_token_index=0):

        if current_sentence_index >= sentences.size():
            return [None for _ in range(num_tokens)]

        sent = sentences.get(current_sentence_index)
        sent_len = sent.getTokens().size()

        if beg_token_index >= sent_len:
            return self.get_tokens_from_right(sentences, num_tokens, current_sentence_index+1)

        tokens = [self.process_token(sent, i) for i in range(beg_token_index, min(beg_token_index+num_tokens, sent_len))]

        num_tokens_left = num_tokens-len(tokens)
        if num_tokens_left == 0:
            return tokens

        return tokens + self.get_tokens_from_right(sentences, num_tokens_left, current_sentence_index+1)

    def get_dependency_path_index_of_path(self, path):

        key = "dependency_path_short"

        try:
            f_index_dict = self.indexed_features[key]
        except KeyError:
            f_index_dict = {}
            self.indexed_features[key] = f_index_dict

        return self.get_feature_value_index(key, path)

    def get_dependency_path_index(self, token_from, position_to):
        path = self.get_dependency_path(token_from, position_to)
        return self.get_dependency_path_index_of_path(path)



    def get_dependency_path(self, token_from, position_to):



        diff_sents = token_from['$sentence_index'] != position_to[0]

        if diff_sents and not self.config['dependency_to_conjuct_in_diff_sents']:
            return None



        incomingEdges = self.doc_temp['dependency_incoming_edges']

        tidx = position_to[1]
        # tidx = token_from['$token_index']
        prev = None
        dep_path = ""
        dep_path_bag = {}
        target_reached = False
        while tidx in incomingEdges:
            if tidx == token_from['$token_index']:
            # if tidx == position_to[1]:
                target_reached = True
                break

            incomingEdge = incomingEdges[tidx]
            source = incomingEdge.getSource()
            if not source:
                break


            deprel = incomingEdge.getLabel(self.doc_temp['dep_col'])

            if diff_sents and deprel in ["root", "conjunct"]:
                # print(dep_path)
                target_reached = True
                break

            if deprel not in dep_path_bag:
                dep_path_bag[deprel] = 1
            else:
                dep_path_bag[deprel] += 1

            if prev is not None:
                if prev != deprel:
                    dep_path = deprel + "-" + dep_path
            else:
                dep_path = deprel


            tidx = source.getIndex()
            prev = deprel

        if not target_reached:
            return None


        if 'dependency_path_short_as_bag' not in self.config or not self.config['dependency_path_short_as_bag']:
            return dep_path

        b = ''
        for k, v in sorted(dep_path_bag.iteritems()): b += k + ','

        return b


    def get_local_sequence(self, sentences, center_position, central_token_type):
        left_tokens = self.get_tokens_from_left(sentences, self.window_size, center_position[0], center_position[1] - 1)

        for i,t in enumerate(left_tokens):
            if not t:
                continue
            t['distance_' + central_token_type] = self.window_size-i
            t['sentence_distance_' + central_token_type] = abs(center_position[0]-t['$sentence_index'])
            t['dependency_path_' + central_token_type] = self.get_dependency_path(t, center_position)

        right_tokens = self.get_tokens_from_right(sentences, self.window_size, center_position[0], center_position[1] + 1)

        for i,t in enumerate(right_tokens):
            if not t:
                continue
            t['distance_' + central_token_type] = i+1
            t['sentence_distance_' + central_token_type] = abs(center_position[0]-t['$sentence_index'])
            t['dependency_path_' + central_token_type] = self.get_dependency_path(t, center_position)



        central = self.process_token(sentences.get(center_position[0]), center_position[1], central_token_type)

        central['distance_'+central_token_type] = 0
        central['sentence_distance_' + central_token_type] = 0
        central['dependency_path_' + central_token_type] = None

        return left_tokens + [central] + right_tokens

    def process_token(self, sentence, tokenIdx, token_type="n/a"):

        if  (sentence.getOrd(), tokenIdx) in self.doc_temp['tokens']:
            copy = self.doc_temp['tokens'][(sentence.getOrd(), tokenIdx)].copy()
            self.map_token_type(copy, token_type)
            return copy

        t = {}

        self.doc_temp['tokens'][(sentence.getOrd(), tokenIdx)] = t
        token = sentence.getTokens().get(tokenIdx)



        if 'dependency' in self.features_to_index:
            # dependency_key = self.doc_temp['sentence_dependency_graphs'][sentence.getOrd()].getTokenNode(tokenIdx + 1).getLabel(7)
            dependency_key = self.doc_temp['sentence_dependency_paths'][sentence.getOrd()][tokenIdx]

            t["dependency_index"] = self.get_feature_value_index("dependency", dependency_key)

        if 'dependency_bag' in self.features_to_index:
            # dependency_key = self.doc_temp['sentence_dependency_graphs'][sentence.getOrd()].getTokenNode(tokenIdx + 1).getLabel(7)
            dependency_key = self.doc_temp['sentence_dependency_path_bag_strs'][sentence.getOrd()][tokenIdx]
            # print(dependency_key)

            t["dependency_bag_index"] = self.get_feature_value_index("dependency_bag", dependency_key)

        if 'annotation' in self.features_to_index:
            chunks_at = sentence.getChunksAt(tokenIdx, self.annotation_types)
            if(len(chunks_at)>0):
                # print([a.getType() for a in chunks_at])

                a_dict ={}
                a_key = ""
                for a in chunks_at:

                    a_type = a.getType()
                    if self.config['map_annotations_enabled']:
                        for a_m in self.config['map_annotations']:
                            if re.match(a_m['from'], a.getType()):
                                a_type = a_m["to"]
                                break


                    if a_type in a_dict:
                        continue

                    a_dict[a_type] = True

                    a_key += a_type
                    if a.getHead() == tokenIdx:
                        a_key += "#head"
                    a_key+="%"
            else:
                a_key = "n/a"

            t["annotation_index"] = self.get_feature_value_index("annotation", a_key)

        self.map_token_type(t, token_type)

        for attr in token.getAttributeIndex().indexes:
            t[attr] = token.getAttributeValue(attr)

        for f in self.features_to_index:
            if f in ['annotation', 'token_type', 'dependency', 'dependency_bag', 'dependency_path_short', 'dependency_path_from', 'dependency_path_to']:
                continue

            t[f + '_index'] = self.get_feature_value_index(f, t[f])

        t['$sentence_index'] = sentence.getOrd()
        t['$token_index'] = tokenIdx

        return t


    def get_feature_value_index(self, feature_name, value):
        f_index_dict = self.indexed_features[feature_name]
        try:
            return f_index_dict[value]
        except KeyError:

            if self.lock_feature_indexes:
                return None

            val_index = len(f_index_dict)
            f_index_dict[value] = val_index
            return val_index



    def map_token_type(self, t, token_type):
        if 'token_type' in self.features_to_index:
            t["token_type_index"] = self.get_feature_value_index("token_type", token_type)

    def init_doc_temp(self):
        self.doc_temp = {
            'tokens': {},
            'sentence_lengths': [],
            'sentence_dependency_graphs': [],
            'sentence_dependency_paths': [],
            'sentence_dependency_path_bags': [],
            'sentence_dependency_path_bag_strs': [],
            'dependency_incoming_edges': {},
            'dep_col': ''
        }

    def process_document(self, document, token_features_generator = None, sequences=None, use_candidate_ratio=True, assign_id_to_annotations=True,
                         dicard_true_rels_over_sentence_distance_limit = True):

        if not token_features_generator:
            token_features_generator = self.get_token_feature_generator()

        if sequences is None:
            sequences = []

        token_features_generator.generateFeatures(document)

        self.init_doc_temp()

        true_rel_dict = {}
        for r in document.getRelations().getRelations():
            rel_type = r.getType()

            matching_rel_configs = [rc for rc in self.config['relations'] if rel_type in rc["types"]]
            if not len(matching_rel_configs):
                continue


            # ann_from = r.getAnnotationFrom()
            # ann_to = r.getAnnotationTo()
            # for a in ann_from.getSentence().getChunksAt(ann_from.getHead()):
            #     print(a.getType())

            relation_position = self.get_relation_position(r)
            # print('true rel: ' + rel_type,relation_position)
            true_rel_dict[relation_position] = r


        matched_true_rels = {}

        for sentence in document.getSentences():
            self.doc_temp['sentence_lengths'].append(sentence.getTokenNumber())
            graph = self.dependency_parser.parse_sentence(sentence)
            dep_col = graph.getDataFormat().getColumnDescription(7)
            self.doc_temp['dep_col'] = dep_col
            self.doc_temp['sentence_dependency_graphs'].append(graph)

            incomingEdges = {}

            for e in graph.getEdges():
                incomingEdges[e.getTarget().getIndex()] = e


            dep_paths = []
            dep_path_bags = []
            dep_path_bag_strs = []
            self.doc_temp['sentence_dependency_paths'].append(dep_paths)
            self.doc_temp['sentence_dependency_path_bags'].append(dep_path_bags)
            self.doc_temp['sentence_dependency_path_bag_strs'].append(dep_path_bag_strs)
            self.doc_temp['dependency_incoming_edges'] = incomingEdges

            for i in range(1, graph.getHighestTokenIndex()+1):

                tidx = i
                prev = None
                dep_path = ""
                dep_path_bag = {}
                while tidx in incomingEdges:
                    incomingEdge = incomingEdges[tidx]
                    source = incomingEdge.getSource()
                    if not source:
                        break

                    deprel = incomingEdge.getLabel(dep_col)
                    if deprel not in dep_path_bag:
                        dep_path_bag[deprel] = 1
                    else:
                        dep_path_bag[deprel] += 1

                    if prev is not None:
                        if prev != deprel:
                            dep_path = deprel + "-" + dep_path
                    else:
                        dep_path = deprel

                    tidx = source.getIndex()
                    prev = deprel
                dep_paths.append(dep_path)
                dep_path_bags.append(dep_path_bag)

                # print(i, dep_path)

                b = ''
                for k, v in  sorted(dep_path_bag.iteritems()): b+=k+','
                dep_path_bag_strs.append(b)

        # print(self.doc_temp['sentence_dependency_path_bag_strs'])
        # exit()

        for rel_conf in self.config['relations']:

            from_candidates = []
            to_candidates = []

            for sentence in document.getSentences():
                for a in self.get_annotations_from_Sentence(sentence):
                    if any(re.match(_from, a.getType()) for _from in rel_conf['from']):
                        from_candidates.append(a)
                    if any(re.match(_from, a.getType()) for _from in rel_conf['to']):
                        to_candidates.append(a)

            print(rel_conf['types'], len(from_candidates), len(to_candidates), 'use_candidate_ratio',use_candidate_ratio )

            sentences = document.getSentences()

            for from_candidate in from_candidates:

                if assign_id_to_annotations and not from_candidate.getId():
                    from_candidate.setId(str(int(round(time.time() * 1000))))

                from_cand_position = self.get_annotation_position(from_candidate)

                from_candidate_seq = self.get_local_sequence(sentences, from_cand_position, 'from')

                from_candidate_seq[self.window_size]['$annotation_id'] = from_candidate.getId()

                for to_candidate in to_candidates:

                    to_candidate_position = self.get_annotation_position(to_candidate)

                    distance = self.get_distance(from_cand_position, to_candidate_position)

                    if distance < 1:
                        continue

                    sent_distance = abs(from_cand_position[0] - to_candidate_position[0])

                    discard_candidate = "candidate_sentence_distance_limit" in rel_conf and rel_conf["candidate_sentence_distance_limit"] is not False and sent_distance > rel_conf["candidate_sentence_distance_limit"]


                    position = (from_cand_position, to_candidate_position)
                    position_reversed = (to_candidate_position, from_cand_position)
                    label = self.irrelevant_class

                    if position in true_rel_dict:
                        relation = true_rel_dict[position]
                        label = relation.getType()
                        matched_true_rels[position] = True
                    elif "allow_reversed" in rel_conf and rel_conf["allow_reversed"] and position_reversed in true_rel_dict:
                        relation = true_rel_dict[position_reversed]
                        label = relation.getType()
                        matched_true_rels[position] = True
                        print("reversed !!!!!!!!!!!", position_reversed)
                    elif discard_candidate or use_candidate_ratio and random.random() > self.config['train_candidate_ratio']:
                        continue

                    if assign_id_to_annotations and not to_candidate.getId():
                        to_candidate.setId(str(int(round(time.time() * 1000))))

                    label_index = self.get_label_index(label)
                    # print(label, position)
                    to_candidate_seq = self.get_local_sequence(sentences, to_candidate_position, 'to')

                    to_candidate_seq[self.window_size]['$annotation_id'] = to_candidate.getId()

                    _from_candidate_seq  = from_candidate_seq
                    left_type = 'from'
                    left_seq = from_candidate_seq
                    left_position = from_cand_position
                    right_type = 'to'
                    right_seq = to_candidate_seq
                    right_position = to_candidate_position

                    # print([t[u'distance_from'] if t else '' for t in from_candidate_seq])
                    # print([t[u'distance_to'] if t else '' for t in to_candidate_seq])
                    # print(self.compare_positions(from_cand_position, to_candidate_position))
                    if self.compare_positions(from_cand_position, to_candidate_position) > 0:
                        left_seq = to_candidate_seq
                        right_seq = from_candidate_seq
                        left_position = to_candidate_position
                        right_position = from_cand_position
                        left_type = 'to'
                        right_type = 'from'

                    left_seq = [t.copy() if t else None for t in left_seq]
                    right_seq = [t.copy() if t else None for t in right_seq]


                    center_idx = self.window_size
                    for i, t in enumerate(left_seq):
                        key = 'distance_' + right_type

                        if not t:
                            continue

                        if key in t:
                            continue

                        t[key] = distance + center_idx - i
                        t['sentence_'+key] = abs(t['$sentence_index'] - right_position[0])

                        t['dependency_path_' + right_type] = self.get_dependency_path(t, right_position)
                        # print(t['orth'], i , t[key])


                    for i, t in enumerate(right_seq):
                        key = 'distance_' + left_type
                        if not t:
                            continue

                        if key in t:
                            continue

                        t[key] = distance - center_idx + i
                        t['sentence_' + key] = abs(t['$sentence_index'] - left_position[0])

                        t['dependency_path_' + left_type] = self.get_dependency_path(t, left_position)


                    if distance <= 2*self.window_size and self.config['no_duplicate_tokens_in_merged_seq']:

                        seq = [t for i,t in enumerate(left_seq) if t and i<self.window_size+distance]

                        seq += [t for i,t in enumerate(right_seq) if t and i >= (self.window_size - max(0, distance-self.window_size-1))]

                        for _ in range(2*self.max_sequence_length - len(seq)):
                            seq.append(None)

                        # print([t[u'orth'] if t else '' for t in seq])

                    else:
                        seq = left_seq+right_seq

                    for t in seq:
                        if not t:
                            continue
                        t['dependency_path_from_index'] = self.get_dependency_path_index_of_path(t['dependency_path_from'])
                        t['dependency_path_to_index'] = self.get_dependency_path_index_of_path(t['dependency_path_to'])


                    # from_type_idx = self.indexed_features["token_type"]['from']
                    # to_type_idx = self.indexed_features["token_type"]['to']
                    # from_token = None
                    # to_token = None
                    # for t in seq:
                    #     if not t:
                    #         continue
                    #     if t['token_type_index'] == from_type_idx:
                    #         from_token = t
                    #     elif t['token_type_index'] == to_type_idx:
                    #         to_token = t
                    #
                    # if not from_token or not to_token:
                    #     print(distance, position)
                    #     print([t['token_type_index'] if t else '' for t in seq])
                    #     print([t[u'distance_from'] if t else '' for t in seq])
                    #     print([t[u'distance_to'] if t else '' for t in seq])
                    #     print([t[u'orth'] if t else '' for t in seq])
                    #
                    #     print([t[u'token_type_index'] if t else '' for t in from_candidate_seq])
                    #     print([t[u'token_type_index'] if t else '' for t in to_candidate_seq])
                    #     print(self.indexed_features["token_type"])
                    #     exit()
                    #     print([t[u'orth'] if t else '' for t in seq])
                    # print([t[u'dependency_path_from'] if t else '' for t in seq])
                    # print([t[u'dependency_path_to'] if t else '' for t in seq])

                    sequences.append((label_index, seq, discard_candidate))


        print('true relations number', len(true_rel_dict), 'matched:', len(matched_true_rels))

        not_matched_rel_count_dict = {}

        if len(true_rel_dict) != len(matched_true_rels):
            for rel_pos in true_rel_dict:
                if rel_pos not in matched_true_rels:
                    rel_type = true_rel_dict[rel_pos].getType()
                    try:
                        count = not_matched_rel_count_dict[rel_type]
                    except KeyError:
                        count = 0
                    not_matched_rel_count_dict[rel_type] = count + 1

        self.doc_temp = {}

        if len(not_matched_rel_count_dict):
            print('not matched relations:')
            print(not_matched_rel_count_dict)

        # exit()
        return sequences, not_matched_rel_count_dict

    def get_label_index(self, label, init_rel_types_if_empty=True):
        if init_rel_types_if_empty and len(self.labels_index) <= 1:
            self.init_rel_types_label_indexes()

        if label in self.labels_index:
            label_index = self.labels_index[label]
        else:
            label_index = len(self.labels_index)
            self.labels_index[label] = label_index

        return label_index

    def init_rel_types_label_indexes(self):
        self.labels_index[self.irrelevant_class] = 0
        for rel_conf in self.config['relations']:
            for t in rel_conf['types']:
                self.get_label_index(t, False)

    def get_annotations_from_Sentence(self, sentence):
        return sentence.getAnnotations(self.annotation_types)


    def get_categorical_embedding_dict(self, values_index, embedding_dim):
        embedding = Sequential()
        embedding.add(Embedding(len(values_index), embedding_dim, input_length=1))

        embedding.compile('rmsprop', 'mse')
        embedding.summary()

        class_embedding_dict = {}

        for c in values_index:
            class_embedding_dict[values_index[c]] = embedding.predict(np.asarray([values_index[c]]))[0][0]

        return class_embedding_dict

    def get_pos_class_embedding_dict(self, classes_index, class_embedding_dim):
        class_embedding = Sequential()
        class_embedding.add(Embedding(len(classes_index), class_embedding_dim, input_length=1))

        class_embedding.compile('rmsprop', 'mse')
        class_embedding.summary()

        class_embedding_dict = {}

        for c in classes_index:
            class_embedding_dict[classes_index[c]] = class_embedding.predict(np.asarray([classes_index[c]]))[0][0]

        return class_embedding_dict

    def get_position_embedding_dict(self, window_size, embedding_dim):
        embedding = Sequential()
        seq_len = window_size * 2 + 1
        embedding.add(Embedding(seq_len, embedding_dim, input_length=1))

        embedding.compile('rmsprop', 'mse')
        embedding.summary()

        embedding_dict = {}

        for p in range(seq_len):
            embedding_dict[p] = embedding.predict(np.asarray([p]))[0][0]

        return embedding_dict

    def save_model(self, model, model_file):
        model.save(model_file)
        self.save_features(model_file + '_ft.pickle')

    def save_features(self, file_name):
        with open(file_name, 'wb') as f:
            pickle.dump({
                'indexed_features':self.indexed_features,
                'labels_index': self.labels_index
            }, f)

    def load_indexed_features(self, indexed_features_file):
        print('loading indexed features from ' + indexed_features_file)
        with open(indexed_features_file, "rb") as fp:
            loaded = pickle.load(fp)
            self.indexed_features = loaded['indexed_features']
            self.labels_index = loaded['labels_index']



    def load_embeddings(self):
        self.input_dim = 0
        for name, e_conf in self.config['w2v_embeddings'].items():
            if not e_conf['enabled']:
                continue

            print("Loading w2v %s embedding ..." % name)
            if name not in self.w2v_embeddings:
                self.w2v_embeddings[name] = Word2Vec.load(e_conf['path'])
            e_dim = self.w2v_embeddings[name].vector_size
            print("w2v %s embedding dim %s." % (name, e_dim))
            self.input_dim += e_dim * len(e_conf['attributes'])
        for name, e_conf in self.config['ft_embeddings'].items():
            if not e_conf['enabled']:
                continue

            print("Loading fasttext %s embedding ..." % name)
            if name not in self.ft_embeddings:
                self.ft_embeddings[name] = FastText.load(e_conf['path'])
            e_dim = self.ft_embeddings[name].vector_size
            print("fasttext %s embedding dim %s." % (name, e_dim))
            self.input_dim += e_dim * len(e_conf['attributes'])

        for attr in self.config['indexed_embeddings']:



            dim = self.config['indexed_embeddings'][attr]
            if attr in ['dependency_path_from', 'dependency_path_to']:
                attr = 'dependency_path_short'


            if not attr in self.indexed_embedding_dicts:
                self.indexed_embedding_dicts[attr] = self.get_categorical_embedding_dict(self.indexed_features[attr],dim)
                print(attr, "indexed embedding dim", dim, 'index_len', len(self.indexed_features[attr]))

            self.input_dim += dim

        if self.config['distance_feature']:
            self.input_dim += 2

        if self.config['sentence_distance_feature']:
            self.input_dim += 2  # sent distance

        # print('position_embedding_dim %s.' % self.config['position_embedding_dim'])
        # self.position_embedding_dict = self.get_position_embedding_dict(self.window_size, self.config['position_embedding_dim'])
        # self.input_dim = self.input_dim + self.config['position_embedding_dim']

    def get_labels_index_inverted(self):
        ind = {}
        for key, value in self.labels_index.iteritems():
            ind[value] = key

        return ind
