from alexnet import create_alexnet
from data import data_generator
from tensorflow.keras.utils import plot_model
import sys

if __name__ == '__main__':
    #getting command line args
    traindir = sys.argv[1]
    validdir = sys.argv[2]
    testdir = sys.argv[3]

    model = create_alexnet()
    plot_model(model, to_file="model.png")

    for it_epoch in range(30):
        for X_train, Y_train in data_generator(traindir, 128):
            acc, mse = model.train_on_batch(X_train, Y_train)
            print("Epoch Number:", it_epoch, "Accuracy:", acc, "Mean Square Error:", mse)





