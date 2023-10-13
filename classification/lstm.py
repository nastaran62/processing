import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import Callback

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dropout, LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers
from tensorflow.keras.layers import  MaxPooling1D, \
    TimeDistributed

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, \
    mean_squared_error, classification_report, accuracy_score, f1_score

class Checkpoint(Callback):

    def __init__(self, test_data, filename):
        self.test_data = test_data
        self.filename = filename

    def on_train_begin(self, logs=None):
        self.fscore = 0.

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        pred_values = self.model.predict(x)
        y_pred = np.argmax(pred_values, axis=1)
        y = np.argmax(y, axis=1)
        precision, recall, f_score, support = \
            precision_recall_fscore_support(y,
                                            y_pred,
                                            average='weighted')


        # Save your model when a better trained model was found
        if f_score > self.fscore:
            self.fscore = f_score
            self.model.save(self.filename, overwrite=True)
            print('********************************************* Higher fscore', f_score, 'found. Save as %s' % self.filename)
        else:
            print("fscore did not improve for ", self.filename, "from ", self.fscore)
        return

def lstm_classification(train_x, train_y, test_x, test_y, scaler=None, model=None):
    model_name = model
    print(train_x.shape, test_x.shape)
    #try:
    # scaling
    if scaler is None:
        scaler = MinMaxScaler()
        scaler.fit(train_x)
    else:
        scaler.partial_fit(train_x)

    # Fit on training set only.
    # Apply transform to both the training set and the test set.
    train_x = scaler.transform(train_x)
    test_x = scaler.transform(test_x)

    train_x = train_x.reshape(train_x.shape[0], train_x.shape[1], 1)
    test_x = test_x.reshape(test_x.shape[0], test_x.shape[1], 1)

    class_weights = \
        class_weight.compute_class_weight(class_weight='balanced',
                                            classes=list(np.unique(train_y)),
                                            y=train_y)
    class_weights = dict(enumerate(class_weights))
    train_y = to_categorical(train_y)
    test_y = to_categorical(test_y)
    print("Preparing classification")
    checkpoint = \
        ModelCheckpoint(model_name,
                        monitor='val_accuracy',
                        verbose=1,
                        save_weights_only=False,
                        mode='max',
                        period=1)
    checkpoint = Checkpoint((np.array(test_x),
                            np.array(test_y)),
                            model_name)
    early_stopping = \
        EarlyStopping(monitor='val_accuracy',
                        patience=10,
                        verbose=1,
                        mode='max')
    reduceLR = ReduceLROnPlateau(monitor='val_accuracy',
                                    factor=0.5,
                                    patience=10,
                                    min_lr=0.0001)
    print("creating model")
    if model_name is None or os.path.exists(model_name) is False:
        model = simple_lstm((train_x.shape[1], train_x.shape[2]),
                            80,  # lstm layers
                            2,  # number of classes
                            dropout=0.5)
        print("model.summary")
        model.summary()
    else:
        model = load_model(model_name)

    print("Start classification")


    history = \
        model.fit(np.array(train_x),
                    np.array(train_y),
                    batch_size=32,
                    epochs=1000,
                    class_weight=class_weights,
                    validation_data=(np.array(test_x),
                                    np.array(test_y)),
                    callbacks=[checkpoint, early_stopping, reduceLR])
    model = load_model(model_name)
    #predictions = model.predict_proba(np.array(test_x))
    # plot history
    #pyplot.plot(history.history['loss'], label='train')
    #pyplot.plot(history.history['val_loss'], label='test')
    #pyplot.legend()
    #pyplot.show()
    pred_values = model.predict(test_x)

    predicted_labels = np.argmax(pred_values, axis=1)
    test_y = np.argmax(test_y, axis=1)

    acc = accuracy_score(predicted_labels, test_y)
    print(classification_report(test_y, predicted_labels))
    precision, recall, f_score, support = \
        precision_recall_fscore_support(test_y,
                                        predicted_labels,
                                        average='weighted')
    '''
    except Exception as error:
        print(error)
        acc=0
        f_score=0
        predicted_labels=0
        predictions=0
    '''
    return model, scaler, acc

def simple_lstm(input_shape, lstm_layers, num_classes, dropout=0.7):
    '''
    Model definition
    '''
    print("Input_shape:", input_shape, " lstm_layers:", lstm_layers, " num_classes: ", num_classes, " dropout:", dropout)
    model = Sequential()
    print("Mdel.add 1")
    model.add(LSTM(lstm_layers, input_shape=input_shape, return_sequences=True))
    #model.add(LSTM(10, return_sequences=True))
    print("model.add 2")
    model.add(LSTM(30))
    #model.add(Dropout(0.4))
    print("model.add 3")
    model.add(Dense(num_classes, activation='softmax'))
    print("compiling model")
    model.compile(loss="categorical_crossentropy",  # categorical_crossentropy
                  optimizer=optimizers.Adam(lr=0.01), metrics=["accuracy"])
    return model
