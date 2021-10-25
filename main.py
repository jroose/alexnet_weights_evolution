
from alexnet import create_alexnet
from tensorflow.keras.utils import plot_model, image_dataset_from_directory
from tensorflow.keras.callbacks import TensorBoard
import sys
import random
import datetime

if __name__ == '__main__':
    random.seed(1)
    #getting command line args
    traindir = sys.argv[1]
    validdir = sys.argv[2]
    testdir = sys.argv[3]

    model = create_alexnet()
    plot_model(model, to_file="model.png")
    Train_ds = image_dataset_from_directory(traindir, image_size= (224, 224), seed=10, labels ="inferred", label_mode = "int" )
    Valid_ds = image_dataset_from_directory(validdir, image_size= (224, 224), seed=10, labels ="inferred", label_mode = "int" )

    #print(Train_ds)
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M")
    tb = TensorBoard(log_dir=str(log_dir), histogram_freq=1)
    model.fit(Train_ds, validation_data=Valid_ds, epochs= 30, callbacks=[tb], batch_size=128)

    # for it_epoch in range(30):
    #     for it_batch, batch in enumerate(data_generator(traindir, 128)):
    #
    #         X_train, Y_train = batch
    #         is_first_batch = it_batch == 0
    #         metrics = model.train_on_batch(X_train, Y_train, reset_metrics = is_first_batch, return_dict = True)
    #         #print("Epoch Number:", it_epoch, "Accuracy:", acc, "Mean Square Error:", mse)
    #
    #         print("Epoch Number:", it_epoch, "Metrics:", metrics)


