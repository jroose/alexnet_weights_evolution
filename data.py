import random

from skimage.io import imread
import pathlib
import numpy as np
#for image_batch in data_generator(128): #image_batch

def data_generator(image_dir, batch_size):
    image_dir = pathlib.Path(image_dir)
    batch = []
    batch_labels = []
    label_ids = {}

    image_paths = list(image_dir.rglob('*.jpg'))
    random.shuffle(image_paths)

    for image_path in image_paths:
        if len(batch) >= batch_size:
            yield (np.stack(batch), np.array(batch_labels, dtype = np.int32))
            batch = []
            batch_labels = []
        label_name = image_path.parent.name
        label_ids.setdefault(label_name, len(label_ids))

        label_id = label_ids[label_name]
        img = imread(str(image_path)) / 255.0
        batch.append(img.astype(np.float32))
        batch_labels.append(label_id)
    if len(batch) > 0:
        yield (np.stack(batch), np.array(batch_labels, dtype=np.int32))
