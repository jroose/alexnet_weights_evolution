from alexnet import create_alexnet

if __name__ == '__main__':
    # import the photos from external library
    # X_input

    # preprocess the photos and data augmentation

    # define layers of network

    X = [] #list of strings
    Y = [] #list of ints, identifying class of image
    #Train, Validation, Test

    X = create_alexnet()
    X.fit(X_Train, Y_Train)

    # Train model, classify images
