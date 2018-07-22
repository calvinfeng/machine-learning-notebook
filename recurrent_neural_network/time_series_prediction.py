import matplotlib.pyplot as plt
import numpy as np
import time
import csv
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM


np.random.seed(1234)


def data_power_consumption(path_to_dataset='datasets/household_power_consumption.txt',
                           sequence_length=50,
                           ratio=1.0):    
    with open(path_to_dataset) as f:
        data = csv.reader(f, delimiter=';')
        power = []
        count = 0
        for line in data:
            try:
                power.append(float(line[2]))
                count += 1
            except ValueError:
                pass
        
            if count / 2049280.0 >= ratio:
                break
        
        # Once all the data are loaded as one large timeseries, we split it into examples. We are 
        # creating a sliding buffer of size 50 here.
        result = []
        for index in range(len(power) - sequence_length):
            result.append(power[index: index + sequence_length])
        result = np.array(result)

        # Center the mean at zero
        mean_val = result.mean()
        result -= mean_val

        # Split data into training and test set
        row = int(round(0.9 * result.shape[0]))
        train = result[:row, :]
        np.random.shuffle(train)
        X_train = train[:, :-1]
        y_train = train[:, -1]
        X_test = result[row:, :-1]
        y_test = result[row:, -1]

        # Reshape data into N by T by D, although D is one-dimensional in this case. LSTM expects 
        # this particular format.
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        return (X_train, y_train, X_test, y_test)


def build_model(input_seq_length=50):
    """
    If return_sequence is true, tehe output will be a sequence of the same length, otherwise the
    output is just one vector.
    """
    layer_dims = [1, 50, 100, 1]
    
    model = Sequential()
    model.add(LSTM(input_shape=(input_seq_length, layer_dims[0]), units=layer_dims[1], return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=layer_dims[2], return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=layer_dims[3]))
    model.add(Activation('linear'))
    
    start = time.time()
    model.compile(loss='mse', optimizer='rmsprop')
    print 'Compilation time:', time.time() - start
    
    return model


def main():
    epochs = 1
    ratio = 1
    seq_len = 50 
    batch_size = 50
    path_to_dataset = 'datasets/household_power_consumption.txt'

    print 'Loading data...'
    X_train, y_train, X_test, y_test = data_power_consumption(path_to_dataset=path_to_dataset,
                                                              sequence_length=seq_len,
                                                              ratio=ratio)
    
    print 'Data is loaded, now compiling model...'
    model = build_model(input_seq_length=seq_len-1)

    print 'Model is compiled, now training...'
    model.fit(x=X_train, y=y_train, batch_size=batch_size, nb_epoch=epochs, validation_split=0.05)
    predicted = model.predict(X_test)
    predicted = np.reshape(predicted, (predicted.size,))

    try:
        fig = plt.figure()
        sub = fig.add_subplot(111)
        sub.plot(y_test[:500])
        plt.plot(predicted[:500])
        plt.show()
    except Exception as e:
        print str(e)


if __name__ == '__main__':
    main()