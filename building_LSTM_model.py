import numpy as np
from keras.models import Sequential
from keras.layers import Activation, TimeDistributed, Dense,LSTM, Bidirectional
from keras.callbacks import Callback, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import preprocessing
import os
from preprocessing import MAXLEN, alphabet

encoder = LSTM(256,input_shape=(MAXLEN, len(alphabet)), return_sequences=True)

decoder=Bidirectional(LSTM(256, return_sequences=True, dropout=0.2))

model=Sequential()
model.add(encoder)
model.add(decoder)
model.add(TimeDistributed(Dense(256)))
model.add(Activation('relu'))
model.add(TimeDistributed(Dense(len(alphabet))))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.001),
              metrics=['accuracy'])

model.summary()

#Spliting training data
train_data, valid_data = train_test_split(preprocessing.list_ngrams, test_size=0.2, random_state=42)

# we have to use data- generation medthod cause this dataset is too large to fit into memory
BATCH_SIZE = 512
def generate_data(data, batch_size):
    cur_index = 0
    while True:
        x, y = [], []
        for i in range(batch_size):  
            y.append(preprocessing.encoder_data(data[cur_index]))
            x.append(preprocessing.encoder_data(preprocessing.add_noise(data[cur_index],0.94,0.985)))
            cur_index += 1
            if cur_index > len(data)-1:
                cur_index = 0
        yield np.array(x), np.array(y)

train_generator = generate_data(train_data, batch_size=BATCH_SIZE)
validation_generator = generate_data(valid_data, batch_size=BATCH_SIZE)

# train the model and save to the Model folder
checkpointer = ModelCheckpoint(filepath=os.path.join('E:\Python\language\Bi-directional-LSTM-Vietnamese-Spelling-AutoCorrection\spelling.h5'), save_best_only=True, verbose=1)

model.fit( train_generator, steps_per_epoch=len(train_data)//BATCH_SIZE, epochs=5,
                    validation_data=validation_generator, validation_steps=len(valid_data)//BATCH_SIZE,
                    callbacks=[checkpointer] )