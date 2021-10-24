from skimage.io import imread
import pathlib
#for image_batch in data_generator(128): #image_batch

def data_generator(image_dir, batch_size):
    image_dir = pathlib.Path(image_dir)

    for image_path in image_dir.rglob('*.jpg'):
        img = imread(str(image_path))
        yield img