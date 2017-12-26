from keras.layers import Input, Flatten, Bidirectional, Dropout, concatenate, GRU, Activation, Conv1D, MaxPooling1D
from keras.layers import LSTM, Dense
from keras.models import Model


# import pickle

def get_model(config, data):
    max_sequence_length = data.max_sequence_length * 2
    input_dims = data.input_dims
    num_classes = data.num_classes

    print("input_dims m", input_dims)

    input_dropout = config['input_dropout']
    lstm_units = config['lstm_units']
    dense_units = config['dense_units']
    filters = int(config['cnn_filters'])  # def 256
    cnn_kernel_size = int(config['cnn_kernel_size'])  # def 5
    pool_size = int(config['pool_size'])  # def 2

    sequence_input = Input(shape=(max_sequence_length, input_dims[0]))
    x = sequence_input
    x = Dropout(input_dropout, input_shape=(max_sequence_length,))(x)
    x = Bidirectional(GRU(lstm_units, return_sequences=True, stateful=False))(x)

    x = Conv1D(filters=filters, kernel_size=cnn_kernel_size, padding='valid', activation='relu')(x)
    x = MaxPooling1D(pool_size=pool_size)(x)

    x = Flatten()(x)


    sequence_input2 = Input(shape=(max_sequence_length, input_dims[0]))
    x2 = sequence_input2
    x2 = Dropout(input_dropout, input_shape=(max_sequence_length,))(x2)
    x2 = Bidirectional(LSTM(lstm_units, return_sequences=True, stateful=False))(x2)

    x2 = Conv1D(filters=256, kernel_size=5, padding='valid', activation='relu')(x2)
    x2 = MaxPooling1D(pool_size=2)(x2)

    x2 = Flatten()(x2)

    x_arr = [x, x2]
    if len(x_arr) > 1:
        x = concatenate(x_arr)
    else:
        x = x_arr[0]

    x = Dropout(0.25)(x)
    if dense_units:
        x = Dense(dense_units)(x)
        x = Activation('relu')(x)


    activation = 'softmax'
    output_units = num_classes
    if config['binary']:
        activation = 'sigmoid'
        output_units = 1

    preds = Dense(output_units, activation=activation)(x)

    inputs = [sequence_input, sequence_input2]

    return Model(inputs, preds)


def get_x(x):
    return [x, x]

def get_x_train(training_data):
    return [training_data['x_train'], training_data['x_train']]

def get_x_test(training_data):
    return [training_data['x_val'], training_data['x_val']]

def get_x_val(training_data):
    return [training_data['x_val'],training_data['x_val']]

def get_x_val_test(test_data):
    return [test_data['x_val_test'],test_data['x_val_test']]


def get_cv_x_test(fold_data):
    return [fold_data['x_test'], fold_data['x_test']]