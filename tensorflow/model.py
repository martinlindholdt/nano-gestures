#!/usr/bin/env python

"""
model.py compiles a Tensorflow light model for use in Aruduino 33 BLE Sense
Reads datafiles in .cvs under ../data/
"""

__status__ = "Prototype"

# using pandas, numpy, matplotlib, tensorflow >2.0.0
# apt-get -qq install xxd
# pip install pandas numpy matplotlib
# pip install tensorflow==2.0.0-rc1

import os
import datetime

import numpy as np
import pandas as pd
import tensorflow as tf

import graphs as g

import files

# Confgigurables -------------------
SHOW = True
SAVE = False
DEBUG = True

# the list of gestures that data is available for
GESTURES = [
    # "punch",
    # "flex",
    "vert",
    "hori"
    # , "circ"
]
BASE_DIR = "../data/"

# Set a fixed random seed value, for reproducibility, this will allow us to get
# the same random numbers each time the notebook is run
SEED = 2548
# SEED = 1337

# Datalines pr sample - set in capture script
SAMPLES_PER_GESTURE = 119

RUN_COUNTER = files.get_counter_and_increment()


# Deferables -------------------

print(f'TensorFlow version = {tf.__version__}\n')


def prepare(GESTURES, SHOW=True, SAVE=False, DEBUG=False):
    if SHOW:
        for gest in GESTURES:
            df = pd.read_csv(BASE_DIR + gest + ".csv")
            g.graph(df, SAVE)

    # create a one-hot encoded matrix that is used in the output
    ONE_HOT_ENCODED_GESTURES = np.eye(len(GESTURES))

    if DEBUG:
        print(ONE_HOT_ENCODED_GESTURES)

    # Will hold the input data in inputs and matching clasification in outputs
    inputs = []
    outputs = []

    # read each csv file and push an input and output
    for gesture_index in range(len(GESTURES)):
        gesture = GESTURES[gesture_index]
        print(f"Processing index {gesture_index} for gesture '{gesture}'.")

        output = ONE_HOT_ENCODED_GESTURES[gesture_index]

        # df = pd.read_csv("/content/" + gesture + ".csv")
        df = pd.read_csv(BASE_DIR + gesture + ".csv")

        # calculate the number of gesture recordings in the file
        num_recordings = int(df.shape[0] / SAMPLES_PER_GESTURE)

        print(f"\tThere are {num_recordings} recordings of the {gesture} gesture.")

        for i in range(num_recordings):
            tensor = []
            for j in range(SAMPLES_PER_GESTURE):
                index = i * SAMPLES_PER_GESTURE + j
                # normalize the input data, between 0 to 1:
                # - acceleration is between: -4 to +4
                # - gyroscope is between: -2000 to +2000
                tensor += [
                    (df['aX'][index] + 4) / 8,
                    (df['aY'][index] + 4) / 8,
                    (df['aZ'][index] + 4) / 8,
                    (df['gX'][index] + 2000) / 4000,
                    (df['gY'][index] + 2000) / 4000,
                    (df['gZ'][index] + 2000) / 4000
                ]

            inputs.append(tensor)
            outputs.append(output)

    # convert the list to numpy array
    inputs = np.array(inputs)
    outputs = np.array(outputs)
    return inputs, outputs


# Pre-process data
inputs, outputs = prepare(GESTURES, False)

print("Total number of gestures: ", len(inputs))
# Now we have a long array with all normalized samples from all files.
# and a outputs array with corresponding classification

print("Data set parsing and preparation complete.")

# Randomize the order of the inputs, so they can be evenly distributed for training, testing, and validation
# https://stackoverflow.com/a/37710486/2020087


np.random.seed(SEED)
tf.random.set_seed(SEED)

num_inputs = len(inputs)
randomize = np.arange(num_inputs)
np.random.shuffle(randomize)
if DEBUG:
    print("Randomize:", randomize)

# Swap the consecutive indexes (0, 1, 2, etc) with the randomized indexes
inputs = inputs[randomize]
outputs = outputs[randomize]


# Split the recordings (group of samples) into three sets: training, testing and validation
# 60% for training, 20% for validation, and 20% for testing.

TRAIN_SPLIT = int(0.6 * num_inputs)
TEST_SPLIT = int(0.2 * num_inputs + TRAIN_SPLIT)

if DEBUG:
    print("# of training cases", TRAIN_SPLIT,
          " and # of test cases", TEST_SPLIT)

inputs_train, inputs_test, inputs_validate = np.split(inputs, [TRAIN_SPLIT, TEST_SPLIT])
outputs_train, outputs_test, outputs_validate = np.split(outputs, [TRAIN_SPLIT, TEST_SPLIT])

if DEBUG:
    print("samples for training:", len(inputs_train))
    print(outputs_train)
    print("samples for testing:", len(inputs_test))
    print(outputs_test)
    print("samples for validation:", len(inputs_validate))
    print(outputs_validate)


print("Data set randomization and splitting complete.")

# Set up TensorBoard logging 
# Enable with `tensorboard --logdir logs`
# point browser to: http://localhost:6006/ 
log_dir = "logs/" + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# build the model and train it
model = tf.keras.Sequential()
# relu is used for performance
model.add(tf.keras.layers.Dense(100, activation='relu'))
# model.add(tf.keras.layers.Dense(50, activation='relu'))
model.add(tf.keras.layers.Dense(15, activation='relu'))
# softmax is used, because we only expect one gesture to occur per input
model.add(tf.keras.layers.Dense(len(GESTURES), activation='softmax'))
model.compile(optimizer='adam',     # or 'sgd' / 'rmsprop' / 'adam' 
              loss='mse',           # mean square error
              metrics=['mae', 'accuracy'])      # Mean absolute error. Or 'accuracy' 

# history = model.fit(inputs_train, outputs_train, epochs=600, batch_size=1, validation_data=(inputs_validate, outputs_validate))
history = model.fit(inputs_train, outputs_train, 
                    epochs=600,
                    batch_size=10,
                    validation_data=(inputs_validate, outputs_validate),
                    callbacks=[tensorboard_callback]
                    )


print("Data model training done")

g.graph_loss(history, 0)
g.graph_mae(history, 0)


# use the model to predict the test inputs
predictions = model.predict(inputs_test)

# print the predictions and the expected ouputs
print("predictions =\n", np.round(predictions, decimals=3))
#print("actual =\n", outputs_test)

print("Predictions")
# predictions_for_print = np.round(predictions, decimals=3)
predictions_for_print = np.round(predictions)
counter = 0
for idx, p in enumerate(predictions_for_print):
    # print (idx, p)
    if (p==outputs_test[idx]).all():
        counter += 1
    print(p, outputs_test[idx], (p==outputs_test[idx]).all())

print("Hit rate:", counter, "/", len(predictions))

if SHOW:
    g.graph_predictions(inputs_test, outputs_test, predictions)


# Convert the model to the TensorFlow Lite format without quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model to disk
open("gesture_model.tflite", "wb").write(tflite_model)
basic_model_size = os.path.getsize("gesture_model.tflite")
print("Model is %d bytes" % basic_model_size)


# # format for arduino

# !echo "const unsigned char model[] = {" > /content/model.h
# !cat gesture_model.tflite | xxd -i      >> /content/model.h
# !echo "};"                              >> /content/model.h

# model_h_size = os.path.getsize("model.h")
# print(f"Header file, model.h, is {model_h_size:,} bytes.")
# print("\nOpen the side panel (refresh if needed). Double click model.h to download the file.")



