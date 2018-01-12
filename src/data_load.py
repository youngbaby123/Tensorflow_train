# -*- coding: utf-8 -*-
import os
from PIL import Image
import cv2
import random
import argparse
from collections import OrderedDict
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def get_files(file_dir, data_root, shuffle):
    # file_dir: data_list.txt
    # return: 乱序后的图片和标签

    data_list = open(file_dir,"r").readlines()
    if shuffle:
        random.shuffle(data_list)

    img_list = []
    label_list = []
    for data_i in data_list:
        img_, label_ = data_i.split()
        img_list.append(os.path.join(data_root, img_))
        label_list.append(label_)

    label_list = [int(i) for i in label_list]

    return img_list, label_list


# img_list,label_list = get_files(file_dir)

def get_batch(image, label, image_W, image_H, batch_size, capacity):
    '''
    Args:
        image: list type
        label: list type
        image_W: image width
        image_H: image height
        batch_size: batch size
        capacity: the maximum elements in queue
    Returns:
        image_batch: 4D tensor [batch_size, width, height, 3], dtype=tf.float32
        label_batch: 1D tensor [batch_size], dtype=tf.int32
    '''

    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)

    # make an input queue
    input_queue = tf.train.slice_input_producer([image, label])

    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels=3)

    ######################################
    # data argumentation should go to here
    ######################################
    # tf.image.resize_images(images, image_W, new_width)
    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)

    # if you want to test the generated batches of images, you might want to comment the following line.
    # 如果想看到正常的图片，请注释掉111行（标准化）和 126行（image_batch = tf.cast(image_batch, tf.float32)）
    # 训练时不要注释掉！
    # image = tf.image.per_image_standardization(image)

    image_batch, label_batch = tf.train.batch([image, label],
                                              batch_size=batch_size,
                                              num_threads=1,
                                              capacity=capacity)

    # you can also use shuffle_batch
    #    image_batch, label_batch = tf.train.shuffle_batch([image,label],
    #                                                      batch_size=BATCH_SIZE,
    #                                                      num_threads=64,
    #                                                      capacity=CAPACITY,
    #                                                      min_after_dequeue=CAPACITY-1)

    label_batch = tf.reshape(label_batch, [batch_size])
    # image_batch = tf.cast(image_batch, tf.float32)
    # image_batch = tf.cast(image_batch, tf.uint8)

    return image_batch, label_batch


def main():
    file_dir = "/home/zkyang/Workspace/task/Tensorflow_task/Tensorflow_train/data/car/train.txt"
    data_root = "/home/zkyang/Workspace/task/Tensorflow_task/Tensorflow_train/data/car/Data_hand"
    shuffle = False
    tra_images, tra_labels = get_files(file_dir, data_root, shuffle)

    BATCH_SIZE = 2
    CAPACITY = 256
    IMG_W = 555
    IMG_H = 555

    image_batch, label_batch = get_batch(tra_images, tra_labels, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
       i = 0
       coord = tf.train.Coordinator()
       threads = tf.train.start_queue_runners(coord=coord)

       try:
           while not coord.should_stop() and i<1:
               img, label = sess.run([image_batch, label_batch])

               # just test one batch
               for j in np.arange(BATCH_SIZE):
                   print('label: %d' %label[j])
                   print img
                   plt.imshow(img[j,:,:,:])
                   plt.show()
               i+=1

       except tf.errors.OutOfRangeError:
           print('done!')
       finally:
           coord.request_stop()
       coord.join(threads)

def single_data_trans():
    file_dir = "/home/zkyang/Workspace/task/Tensorflow_task/Tensorflow_train/data/car/train.txt"
    data_root = "/home/zkyang/Workspace/task/Tensorflow_task/Tensorflow_train/data/car/Data_hand"
    shuffle = False

    tra_images, tra_labels = get_files(file_dir, data_root, shuffle)
    img_path = tra_images[1]
    label_ = tra_labels[1]
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # img_ = tf.gfile.GFile(img_path, 'rb').read()
    img_ = tf.read_file(img_path)
    with tf.Session(config=config) as sess:
        img_data = tf.image.decode_jpeg(img_)
        plt.subplot(231)
        plt.title("ori")
        plt.imshow(img_data.eval())
        # plt.show()

        resized = tf.image.resize_images(img_data, [300, 300], method=0)  # 第一个参数为原始图像，第二个参数为图像大小，第三个参数给出了指定的算法
        resized = np.asarray(resized.eval(), dtype='uint8')
        plt.subplot(232)
        plt.title("ori")
        plt.imshow(resized)
        # plt.show()

        croped = tf.image.resize_image_with_crop_or_pad(img_data, 200, 200)  # 目标图像大小<原始图像的大小，则截取原始图像的居中部分，
        padded = tf.image.resize_image_with_crop_or_pad(img_data, 800, 800)  # 目标图像大小>原始图像的大小，则会在原始图像的四周填充全为0背景
        plt.subplot(233)
        plt.title("ori")
        plt.imshow(croped.eval())
        # plt.show()
        plt.subplot(234)
        plt.title("ori")
        plt.imshow(padded.eval())
        # plt.show()

        central_cropped = tf.image.central_crop(img_data, 0.5)  # 按照比例裁剪图像，第二个参数为调整比例，比例取值[0,1]
        plt.subplot(235)
        plt.title("ori")
        plt.imshow(central_cropped.eval())
        plt.show()



if __name__ == '__main__':
    # main()
    single_data_trans()
