import os
from PIL import Image
from PIL import ImageStat
import numpy as np

train_images_dir = 'C:\\Users\Sergio\PycharmProjects\PlateRecognition\\train_images\cn_character\\'
test_images_dir = 'C:\\Users\Sergio\PycharmProjects\PlateRecognition\\test_images\cn_character\\'

labels = ['anhui', 'beijing', 'chongqing', 'fujian', 'gansu', 'guangdong',
          'guangxi', 'guizhou', 'hainan', 'hebei', 'heilongjiang', 'henan',
          'hubei', 'hunan', 'jiangsu', 'jiangxi', 'jilin', 'liaoning',
          'neimenggu', 'ningxia', 'qinghai', 'shandong', 'shanghai', 'shannxi',
          'shanxi', 'sichuan', 'tianjin', 'xinjiang', 'xizang', 'yunnan', 'zhejiang']
label2code = {'anhui': 0, 'beijing': 1, 'chongqing': 2, 'fujian': 3, 'gansu': 4,
              'guangdong': 5, 'guangxi': 6, 'guizhou': 7, 'hainan': 8, 'hebei': 9,
              'heilongjiang': 10, 'henan': 11, 'hubei': 12, 'hunan': 13, 'jiangsu': 14,
              'jiangxi': 15, 'jilin': 16, 'liaoning': 17, 'neimenggu': 18, 'ningxia': 19,
              'qinghai': 20, 'shandong': 21, 'shanghai': 22, 'shannxi': 23, 'shanxi': 24,
              'sichuan': 25, 'tianjin': 26, 'xinjiang': 27, 'xizang': 28, 'yunnan': 29, 'zhejiang': 30}

IMAGE_WIDTH = 16
IMAGE_HEIGHT = 28
IMAGE_RATIO = IMAGE_HEIGHT / IMAGE_WIDTH


def convert2bin(image, show=False):

    # reshape
    """width, height = image.size
    if float(height) / width > IMAGE_RATIO:
        y = (height - width * IMAGE_RATIO) / 2
        image = image.crop((0, y, width, height - y))
    elif float(height) / width < 2:
        x = (width - height / IMAGE_RATIO) / 2
        image = image.crop((x, 0, width - x, height))"""

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
            for _l in range(31 - label2code[label] - 1):
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
            for _l in range(31 - label2code[label] - 1):
                label_array.append(0)

            batch_labels.append(np.array(label_array, dtype=float))

    return batch_images, batch_labels
