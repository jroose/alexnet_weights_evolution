from alexnet import create_alexnet
from tensorflow.data.Dataset import load, from_generator
from data import data_generator
import sys

#courtesy of Angel Igareta
#https://towardsdatascience.com/how-to-split-a-tensorflow-dataset-into-train-validation-and-test-sets-526c8dd29438
def get_dataset_partitions_tf(ds, ds_size, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True,
                              shuffle_size=10000):
    assert (train_split + test_split + val_split) == 1

    if shuffle:
        # Specify seed to always have the same split distribution between runs
        ds = ds.shuffle(shuffle_size, seed=12)

    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)

    train_ds = ds.take(train_size)
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)

    return train_ds, val_ds, test_ds

if __name__ == '__main__':
    #getting command line args
    traindir = sys.argv[1]
    validdir = sys.argv[2]
    testdir = sys.argv[3]

    model = create_alexnet()

    for it_epoch in range(30):
        for X_train, Y_train in data_generator(traindir, 128):
            acc, mse = model.train_on_batch(X_train, Y_train)
            print("Epoch Number:", it_epoch, "Accuracy:", acc, "Mean Square Error:", mse)





