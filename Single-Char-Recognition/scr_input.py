import os
from PIL import Image
from PIL import ImageStat
import numpy as np

train_images_dir = 'C:\\Users\Sergio\PycharmProjects\PlateRecognition\\train_images\\'
test_images_dir = 'C:\\Users\Sergio\PycharmProjects\PlateRecognition\\test_images\\'

labels = ['0',  '1',  '2',  '3',  '4',  '5',  '6',  '7',  '8',  '9',
          'A',  'B',  'C',  'D',  'E',  'F',  'G',  'H',  'J',  'K',
          'L',  'M',  'N',  'P',  'Q',  'R',  'S',  'T',  'U',  'V',
          'W',  'X',  'Y',  'Z']
label2code = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
             'A': 10,  'B': 11,  'C': 12,  'D': 13,  'E': 14,  'F': 15,  'G': 16,  'H': 17,  'J': 18,
             'K': 19,  'L': 20,  'M': 21,  'N': 22,  'P': 23,  'Q': 24,  'R': 25,  'S': 26,  'T': 27,
             'U': 28,  'V': 29,  'W': 30,  'X': 31,  'Y': 32,  'Z': 33}

IMAGE_WIDTH = 12
IMAGE_HEIGHT = 24


def convert2bin(image, show=False):

    # reshape
    width, height = image.size
    if float(height) / width > 2:
        y = (height - width * 2) / 2
        image = image.crop((0, y, width, height - y))
    elif float(height) / width < 2:
        x = (width - height / 2) / 2
        image = image.crop((x, 0, width - x, height))

    image = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.ANTIALIAS)

    # convert to binary value
    threshold = ImageStat.Stat(image).mean[0]
    array = np.array(image, dtype=np.int).reshape(IMAGE_HEIGHT * IMAGE_WIDTH)

    bin_array = []
    for pixel in array:
        if pixel > threshold:
            bin_array.append(255)
        else:
            bin_array.append(0)

    if show:
        Image.fromarray(np.array(bin_array).reshape(IMAGE_HEIGHT, IMAGE_WIDTH)).show()

    return np.multiply(np.array(bin_array, dtype=np.float), 1.0 / 255.0)


def train_inputs():
    batch_images = []
    batch_labels = []
    print('Generate train data...')
    for label in labels:
        file_list = os.listdir(train_images_dir + label)

        for i in file_list:
            curr_name = train_images_dir + label + "\\" + i

            curr_image = Image.open(curr_name).convert('L')

            batch_images.append(convert2bin(curr_image))

            # generate one_hot vector
            label_array = []
            for _l in range(label2code[label]):
                label_array.append(0)
            label_array.append(1)
            for _l in range(34 - label2code[label] - 1):
                label_array.append(0)

            batch_labels.append(np.array(label_array, dtype=float))

    return batch_images, batch_labels


def test_inputs():
    batch_images = []
    batch_labels = []
    print('Generate test data...')
    for label in labels:
        file_list = os.listdir(test_images_dir + label)

        for i in file_list:
            curr_name = test_images_dir + label + "\\" + i

            curr_image = Image.open(curr_name).convert('L')
            batch_images.append(convert2bin(curr_image))

            # generate one_hot vector
            label_array = []
            for _l in range(label2code[label]):
                label_array.append(0)
            label_array.append(1)
            for _l in range(34 - label2code[label] - 1):
                label_array.append(0)

            batch_labels.append(np.array(label_array, dtype=float))

    return batch_images, batch_labels
