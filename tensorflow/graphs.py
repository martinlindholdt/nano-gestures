#!/usr/bin/env python

"""
graphs.py for plotting Tensorflow data for use in Aruduino 33 BLE Sense
"""

import matplotlib.pyplot as plt
from files import get_counter_and_increment

STORE = False
COUNTER = 0


def store_graphs(store=True):
    global STORE, COUNTER
    STORE = store
    if STORE:
        COUNTER = get_counter_and_increment()


def graph(df):
    # no of data entries
    index = range(1, len(df['aX']) + 1)

    plt.rcParams["figure.figsize"] = (20, 10)

    plt.plot(index, df['aX'], 'g.', label='x', linestyle='solid', marker=',')
    plt.plot(index, df['aY'], 'b.', label='y', linestyle='solid', marker=',')
    plt.plot(index, df['aZ'], 'r.', label='z', linestyle='solid', marker=',')
    plt.title("Acceleration")
    plt.xlabel("Sample #")
    plt.ylabel("Acceleration (G)")
    plt.legend()
    if STORE:
        plt.savefig("graphs/{}-rawdata-accel.png".format(COUNTER))
    plt.show()

    plt.plot(index, df['gX'], 'g.', label='x', linestyle='solid', marker=',')
    plt.plot(index, df['gY'], 'b.', label='y', linestyle='solid', marker=',')
    plt.plot(index, df['gZ'], 'r.', label='z', linestyle='solid', marker=',')
    plt.title("Gyroscope")
    plt.xlabel("Sample #")
    plt.ylabel("Gyroscope (deg/sec)")
    plt.legend()
    if STORE:
        plt.savefig("graphs/{}-rawdata-gyro.png".format(COUNTER))
    plt.show()
    return


def graph_loss(history, SKIP=0):
    # increase the size of the graphs. The default size is (6,4).
    plt.rcParams["figure.figsize"] = (20, 10)

    # graph the loss, the model above is configure to use "mean squared error" as the loss function
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)

    plt.plot(epochs[SKIP:], loss[SKIP:], 'g.', label='Training loss')
    plt.plot(epochs[SKIP:], val_loss[SKIP:], 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    if STORE:
        plt.savefig("graphs/{}-loss.png".format(COUNTER))
    plt.show()
    print(plt.rcParams["figure.figsize"])


# mea = mean absolute error
def graph_mae(history, SKIP=0):
    # graph of mean absolute error
    mae = history.history['mae']
    val_mae = history.history['val_mae']
    epochs = range(1, len(mae) + 1)

    plt.plot(epochs[SKIP:], mae[SKIP:], 'g.', label='Training MAE')
    plt.plot(epochs[SKIP:], val_mae[SKIP:], 'b.', label='Validation MAE')
    plt.title('Training and validation mean absolute error')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()
    if STORE:
        plt.savefig("graphs/{}-mean-absolute-error.png".format(COUNTER))
    plt.show()


def graph_predictions(inputs_test, outputs_test, predictions):
    # Plot the predictions along with to the test data
    plt.clf()
    plt.title('Training data predicted vs actual values')
    plt.plot(inputs_test, outputs_test, 'b.', label='Actual')
    plt.plot(inputs_test, predictions, 'r.', label='Predicted')
    if STORE:
        plt.savefig("graphs/{}-predictions.png".format(COUNTER))
    plt.show()
