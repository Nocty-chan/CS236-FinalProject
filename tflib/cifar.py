import numpy as np

import os
import urllib
import gzip
import pickle

def unpickle(file, mode="cifar10"):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    if mode == "cifar10":
      return dict[b'data'], dict[b'labels']
    else:
      return dict[b'data'], dict[b'fine_labels']

def unpickle_labels(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict[b'fine_label_names']

def cifar_generator(filenames, batch_size, data_dir, mode="cifar10", classes=None):
    all_data = []
    all_labels = []
    for filename in filenames:
        data, labels = unpickle(data_dir + '/' + filename, mode)
        all_data.append(data)
        all_labels.append(labels)
    names = unpickle_labels(data_dir + '/meta')
    images = np.concatenate(all_data, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    if classes is not None:
        subset_labels = (labels // 10 == classes) # Only take 10 classes from cifar100
        labels = labels[subset_labels] - classes * 10 # Re-center labels between 0 and 9
        images = images[subset_labels]
        names = names[classes * 10: (classes + 1)*10]


    print("[*] Loading classes\n ", "\n".join(map(str, names)))
    print("[*] Num examples ", len(images))
    def get_epoch():
        rng_state = np.random.get_state()
        np.random.shuffle(images)
        np.random.set_state(rng_state)
        np.random.shuffle(labels)

        for i in range(int(len(images) / batch_size)):
            yield (images[i*batch_size:(i+1)*batch_size], labels[i*batch_size:(i+1)*batch_size])

    return get_epoch


def load_cifar10(batch_size, data_dir):
    return (
        cifar_generator(['data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5'], batch_size, data_dir),
        cifar_generator(['test_batch'], batch_size, data_dir)
    )

def load_cifar100(batch_size, data_dir, classes=None):
    return (
        cifar_generator(['train'], batch_size, data_dir, mode="cifar100", classes=classes),
        cifar_generator(['test'], batch_size, data_dir, mode="cifar100", classes=classes)
    )
