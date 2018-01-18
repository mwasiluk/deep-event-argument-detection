from __future__ import print_function

import json
import random as rn
import sys
# import pickle
from cStringIO import StringIO
from optparse import OptionParser

import gc
import keras
import numpy as np
import tensorflow as tf
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from sklearn.utils import class_weight

from data import Data
from models import branched_bi_gru_lstm, bi_gru, bi_gru_cnn, bi_gru_2cnn, branched_bi_gru_lstm_cnn, cnn_bi_lstm, bi_lstm_cnn
from liner2 import create_annotation, get_writer, create_relation

from utils import print_stats, mkdir_p, merge_several_folds_results, print_p_r_f, sliding_window
from db import DatabaseMng

IRRELEVANT_CLASS = 'n/a'

default_net_model = bi_gru_cnn

MODEL_DICT = {
    "bi_gru": bi_gru,
    "bi_gru_cnn": bi_gru_cnn,
    "bi_lstm_cnn": bi_lstm_cnn,
    "bi_gru_2cnn": bi_gru_2cnn,
    "cnn_bi_lstm": cnn_bi_lstm,
    "branched_bi_gru_lstm": branched_bi_gru_lstm,
    "branched_bi_gru_lstm_cnn": branched_bi_gru_lstm_cnn
}


def run(config, input_files_index = None, output_file=None, token_sequences_file_name=None, file_to_save_token_sequences=None, model_file=None, train=False, cv=False, input_format="batch:ccl", output_format="batch:ccl"):

    set_seed(config['seed'])

    window_size = config['window_size']
    db = None
    if "db" in config and config["db"]["enabled"]:
        db_conf = config["db"]
        db = DatabaseMng(dbname=db_conf["dbname"], table_name=db_conf["table_name"], host=db_conf["host"], user=db_conf["user"], password=db_conf["password"])
        db.init()

    print('window_size',window_size)

    data = Data(config=config, input_files_index = input_files_index, window_size=window_size,
                 irrelevant_class = IRRELEVANT_CLASS, single_class=config['single_class'], token_sequences_file_name=token_sequences_file_name,
                 file_to_save_token_sequences=file_to_save_token_sequences, cv=cv, input_format=input_format)

    if train:
        run_train(config, data, model_file, db=db)
    elif cv:
        run_cv(config, data, model_file, db=db)
    else:
        run_pipe(config, data, model_file, output_file=output_file, output_format=output_format, db=db)


def set_seed(seed):
    # there is some problem with seeding https://github.com/fchollet/keras/issues/2280
    np.random.seed(seed)
    tf.set_random_seed(seed)
    rn.seed(seed)

def get_net_model(config):
    if "model_name" not in config:
        print("model_name not set in config, using default")
        return default_net_model

    model_name = config["model_name"]

    try:
        return MODEL_DICT[model_name]
    except KeyError:
        print("unknown model name: %s, using default" % model_name)
        return default_net_model



def run_pipe(config, data, model_file, output_file, output_format, db):


    print("pipe mode")
    model = keras.models.load_model(model_file)
    model.summary()

    data.load_indexed_features(model_file + '_ft.pickle')
    data.load_embeddings()
    reader = data.get_reader()
    writer = get_writer(output_file, output_format)

    print("input dims", data.input_dims)

    labels_index_inverted = {}
    for key, value in data.labels_index.iteritems():
        labels_index_inverted[value] = key


    from_type_idx = data.indexed_features["token_type"]['from']
    to_type_idx = data.indexed_features["token_type"]['to']

    net_model = get_net_model(config)

    while True:
        document = reader.nextDocument()
        if document is None:
            break

        sequences = []
        data.process_document(document, sequences=sequences, use_candidate_ratio=False)
        if not len(sequences):
            continue
        x = [data.get_vector_sequence(seq) for seq in sequences]

        y_pred = model.predict(net_model.get_x(np.asarray(x)))
        y_pred = map(lambda v: labels_index_inverted[v.argmax()], y_pred)

        sentences = document.getSentences()

        for idx, l in enumerate(y_pred):
            if l == data.irrelevant_class:
                continue
            from_token = None
            to_token = None
            for t in sequences[idx][1]:
                if not t:
                    continue
                if t['token_type_index'] == from_type_idx:
                    from_token = t
                elif t['token_type_index'] == to_type_idx:
                    to_token = t
            # if not from_token or not to_token:
            #     continue
            # print(l,from_token, to_token)
            annotation_from = [a for a in sentences.get(from_token['$sentence_index']).getChunksAt(from_token['$token_index']) if a.getId() == from_token['$annotation_id']][0]
            annotation_to = [a for a in sentences.get(to_token['$sentence_index']).getChunksAt(to_token['$token_index']) if a.getId() == to_token['$annotation_id']][0]
            print(l, from_token[u'orth'], to_token[u'orth'])

            relation = create_relation(annotation_from, annotation_to, l)
            document.addRelation(relation)



        writer.writeDocument(document)

    reader.close()
    writer.close()

    # model.predict(x_test)


def run_cv(config, data, model_file=None, db=None):

    fold_idx = 0
    prf_results = []
    p_r_f_avg_micro_results = []
    p_r_f_avg_weighted_results = []
    acc_list = []

    net_model = get_net_model(config)

    for fold_data in data.get_cv_folds_training_data(config['validation_split']):
        print("Running Fold", fold_idx + 1)
        fold_idx += 1
        model = net_model.get_model(config, data)

        sys.stdout = mystdout = StringIO()
        model.summary()
        sys.stdout = sys.__stdout__
        model_summary = mystdout.getvalue()

        train_model(model, config, fold_data)

        test_reader = fold_data['test_reader']
        fold_data = None
        gc.collect()

        test_data = data.get_cv_fold_test_data(test_reader)


        x_test = net_model.get_cv_x_test(test_data)
        y_pred = model.predict(x_test)

        y_pred = get_y_pred(config, y_pred)

        y_test = test_data['y_test']


        y_test = y_test.tolist()
        y_pred = y_pred.tolist()
        for l in data.labels_index:
            y_test.append(data.get_label_from_idx(data.labels_index[l]))
            y_pred.append(data.get_label_from_idx(data.labels_index[data.irrelevant_class]))

        y_test = np.array(y_test)
        y_pred = np.array(y_pred)
        p_r_f, acc, p_r_f_avg_micro, p_r_f_avg_weighted = print_stats(y_test, y_pred, data.labels_index, config['binary'])

        not_matched = test_data['not_matched']

        if len(not_matched):
            print()
            print('Computing metrics with not matched true candidates')
            y_test = y_test.tolist()
            y_pred = y_pred.tolist()
            for label in not_matched:
                for _ in range(not_matched[label]):
                    y_test.append(data.get_label_from_idx(data.labels_index[label]))

                    y_pred.append(data.get_label_from_idx(data.labels_index[data.irrelevant_class]))

            p_r_f, acc, p_r_f_avg_micro, p_r_f_avg_weighted = print_stats(np.array(y_test), np.array(y_pred),
                                                                          data.labels_index, config['binary'])



        acc_list.append(acc)
        prf_results.append(p_r_f)
        p_r_f_avg_micro_results.append(p_r_f_avg_micro)
        p_r_f_avg_weighted_results.append(p_r_f_avg_weighted)

        if model_file:
            data.save_model(model, model_file+"_fold_"+str(fold_idx))

    cv_prf = merge_several_folds_results(prf_results)
    cv_p_r_f_avg_micro = merge_several_folds_results(p_r_f_avg_micro_results)
    p_r_f_avg_weighted_prf = merge_several_folds_results(p_r_f_avg_weighted_results)

    print_p_r_f(cv_prf, data.labels_index)

    print('micro averaged:')
    print(cv_p_r_f_avg_micro)

    print('weighted averaged:')
    print(p_r_f_avg_weighted_prf)

    if db:
        db.save_result(config, np.mean(acc_list), cv_prf, model_summary, data.num_classes, np.array([]), labels_index=data.labels_index,
                       model_id=model_file, p_r_f_avg_micro = cv_p_r_f_avg_micro, p_r_f_avg_weighted = p_r_f_avg_weighted_prf)

def get_y_pred(config, y_pred):
    if config['binary']:
        y_pred = map(lambda v: v > 0.5, y_pred)
    return y_pred


def run_train(config, data, model_file, db=None):
    net_model = get_net_model(config)
    data.load()
    training_data = data.get_training_data(config['validation_split'])
    model = net_model.get_model(config, data)
    model.summary()

    gc.collect()
    tr = train_model(model, config, training_data)

    training_data = None
    gc.collect()

    test_data = data.get_validation_test_data(config['validation_split'])



    x_test = net_model.get_x_val_test(test_data)
    y_test = test_data['y_val_test']
    y_pred = model.predict(x_test)
    y_pred = get_y_pred(config, y_pred)

    p_r_f, acc, p_r_f_avg_micro, p_r_f_avg_weighted = print_stats(y_test, y_pred, data.labels_index, config['binary'])
    if model_file:
        data.save_model(model, model_file)


    not_matched = test_data['not_matched']

    if len(not_matched):
        print()
        print ('Computing metrics with not matched true candidates')
        y_test = y_test.tolist()
        y_pred = y_pred.tolist()
        for label in not_matched:
            for _ in range(not_matched[label]):
                y_test.append(data.get_label_from_idx(data.labels_index[label]))

                y_pred.append(data.get_label_from_idx(data.labels_index[data.irrelevant_class]))

        p_r_f, acc, p_r_f_avg_micro, p_r_f_avg_weighted = print_stats(np.array(y_test), np.array(y_pred), data.labels_index, config['binary'])


    sys.stdout = mystdout = StringIO()
    model.summary()
    sys.stdout = sys.__stdout__
    model_summary = mystdout.getvalue()


    if db:
        db.init()
        db.save_result(config, acc, p_r_f, model_summary, data.num_classes, tr["class_weights"], labels_index=data.labels_index,
                       model_id=model_file, p_r_f_avg_micro = p_r_f_avg_micro, p_r_f_avg_weighted = p_r_f_avg_weighted)

def train_model(model, config, training_data):
    net_model = get_net_model(config)
    print('Training model.')
    # compute balanced class weights
    if config['binary']:
        class_w = class_weight.compute_class_weight('balanced', np.array([0, 1]), training_data['y_train'])
    else:
        y_train_true = map(lambda v: v.argmax(), training_data['y_train'])
        class_w = class_weight.compute_class_weight('balanced', np.unique(y_train_true), y_train_true)
    print("Computed class weight:")
    print(class_w)

    saved_model_checkpoints_dir = '../saved_model_checkpoints'
    mkdir_p(saved_model_checkpoints_dir)

    mcp = ModelCheckpoint('%s/weights.best.hdf5' % saved_model_checkpoints_dir, monitor="val_acc",
                          save_best_only=True, save_weights_only=False, verbose=1)

    loss = 'categorical_crossentropy'
    if config['binary']:
        loss = 'binary_crossentropy'

    train_inputs = net_model.get_x_train(training_data)
    val_inputs = net_model.get_x_val(training_data)

    model.compile(loss=loss,
                  optimizer=optimizers.Adam(lr=config['lr'], decay=config['lr_decay']),
                  metrics=['accuracy'])

    model.fit(train_inputs, training_data['y_train'], shuffle=False,
              batch_size=config['batch_size'],
              epochs=config['epochs'],
              validation_data=(val_inputs, training_data['y_val']),
              class_weight=class_w,
              callbacks=[mcp])

    model.load_weights("%s/weights.best.hdf5" % saved_model_checkpoints_dir)
    return {
        "class_weights": class_w
    }

def load_config(filename):
    with open(filename) as json_data:
        c = json.load(json_data)
    return c


def go():
    parser = OptionParser(usage="Tool for event detection")
    parser.add_option('-c', '--config', type='string', action='store',
                      dest='config',
                      help='json config file location')
    parser.add_option('-t', '--train', action='store_true', default=False,
                      dest='train',
                      help='train model')
    parser.add_option('-e', '--eval', action='store_true', default=False,
                      dest='eval',
                      help='train and eval model using cv')
    parser.add_option('-m', '--model', type='string', action='store',
                      dest='model',
                      help='model file location')
    parser.add_option('-i', '--input-format', type='string', action='store',
                      dest='input_format',
                      help='input format', default="batch:ccl")
    parser.add_option('-o', '--output-format', type='string', action='store',
                      dest='output_format',
                      help='output format, default same as input format', default=None)
    parser.add_option('--file-to-save-token-sequences', type='string', action='store',
                      dest='file_to_save_token_sequences',
                      help='file to save processed token sequences (not vectors), useful for training', default=None)
    parser.add_option('--token-sequences-file', type='string', action='store',
                      dest='token_sequences_file_name',
                      help='file with saved processed token sequences (not vectors), useful for training', default=None)
    (options, args) = parser.parse_args()
    fn_output = None
    if options.train or options.eval:
        if len(args) != 1:
            print('Need to provide input')

            print('See --help for details.')
            sys.exit(1)
        fn_input = args[0]
    else:

        if len(args) != 2:
            print('Need to provide input and output.')
            print('See --help for details.')
            sys.exit(1)
        fn_input, fn_output = args

    config = load_config(options.config)

    if not options.output_format:
        options.output_format = options.input_format


    run(config, fn_input, fn_output, model_file=options.model, train=options.train, cv = options.eval, input_format=options.input_format, output_format = options.output_format,
         file_to_save_token_sequences = options.file_to_save_token_sequences, token_sequences_file_name=options.token_sequences_file_name)

if __name__ == '__main__':
    go()