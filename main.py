from alexnet import create_alexnet
from tensorflow.keras.utils import plot_model, image_dataset_from_directory
import sys
import random

if __name__ == '__main__':
    random.seed(1)
    #getting command line args
    traindir = sys.argv[1]
    validdir = sys.argv[2]
    testdir = sys.argv[3]

    model = create_alexnet()
    plot_model(model, to_file="model.png")
    X_train, Y_train = image_dataset_from_directory(traindir, image_size= (224, 224), seed=10)
    model.fit(X_train, Y_train, epochs= 30)

    # for it_epoch in range(30):
    #     for it_batch, batch in enumerate(data_generator(traindir, 128)):
    #
    #         X_train, Y_train = batch
    #         is_first_batch = it_batch == 0
    #         metrics = model.train_on_batch(X_train, Y_train, reset_metrics = is_first_batch, return_dict = True)
    #         #print("Epoch Number:", it_epoch, "Accuracy:", acc, "Mean Square Error:", mse)
    #
    #         print("Epoch Number:", it_epoch, "Metrics:", metrics)


