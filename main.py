from alexnet import create_alexnet
#from tensorflow.data.Dataset import load


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
    # import the photos from external library
    # X_input

    # preprocess the photos and data augmentation

    # define layers of network

    X = [] #list of strings
    Y = [] #list of ints, identifying class of image
    #Train, Validation, Test

    X_Train,  X_Valid, X_Test =
    Y_Train, Y_Valid,  Y_Test =

    X = create_alexnet()
    X.fit(X_Train, Y_Train)

    # Train model, classify images
