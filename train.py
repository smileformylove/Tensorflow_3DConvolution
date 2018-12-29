import tensorflow as tf
import numpy as np

import cv2

import os
import glob

from random import shuffle

from Define import *
from conv3D import *
from utils import *

if __name__ == '__main__':
    
    root_db_dir = '../../DB/UCF/'
    label_names = os.listdir(root_db_dir)

    train_data = []
    test_data = []

    for label_name in label_names:
        label = LABEL_DIC[label_name]
        video_paths = glob.glob(root_db_dir + label_name + '/*')

        paths = []
        for video_path in video_paths:
            paths.append([video_path, label])
    
        length = len(video_paths)
        train_data += paths[:int(0.7 * length)]
        test_data += paths[int(0.7 * length):]

    print('train :', len(train_data))
    print('test :', len(test_data))

    #path define
    model_path = './model/'
    model_name = '3d_conv_{}.ckpt'
    
    #model build
    input_var = tf.placeholder(tf.float32, shape=[None, SEQUENCE, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNEL], name = 'input')
    label_var = tf.placeholder(tf.float32, shape=[None, CLASSES], name = 'label')

    conv3d = Convolution_3D(input_var)

    learning_rate = tf.placeholder(tf.float32, name = 'learning_rate')
    
    #loss
    ce = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = label_var, logits = conv3d))

    #optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate, epsilon = 1e-4)
    train = optimizer.minimize(ce)

    #accuracy
    correct_prediction = tf.equal(tf.argmax(conv3d, 1), tf.argmax(label_var, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    #save
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        #saver.restore(sess, model_path + model_name.format(99))

        #epoch
        print('set batch size :', BATCH_SIZE)

        #epoch_learning_rate = 1e-1 #momentum
        epoch_learning_rate = 1e-4
        for epoch in range(1, MAX_EPOCHS+1):
            if epoch == (MAX_EPOCHS * 0.5) or epoch == (MAX_EPOCHS * 0.75):
                epoch_learning_rate /= 10
            
            #init
            shuffle(train_data)
            list_input_var = []
            list_label_var = []

            train_cnt = 0
            train_acc = 0.0
            train_loss = 0.0

            #train
            for _train_data in train_data:
                train_path, train_label = _train_data

                frames = []

                video = cv2.VideoCapture(train_path)
                while True:
                    ret, frame = video.read()
                    if not ret:
                        break

                    frame = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation = cv2.INTER_CUBIC)
                    frame = np.asarray(frame, dtype = np.float32)
                    frame /= 255.0

                    frames.append(frame)

                    if len(frames) == SEQUENCE:
                        break

                if len(frames) != SEQUENCE:
                    continue

                label = one_hot(train_label, CLASSES)

                list_input_var.append(frames.copy())
                list_label_var.append(label.copy())

                if len(list_input_var) == BATCH_SIZE:
                    np_input_var = np.asarray(list_input_var, dtype = np.float32)
                    np_label_var = np.asarray(list_label_var, dtype = np.float32)

                    input_map = { input_var : np_input_var,
                                  label_var : np_label_var,
                                  learning_rate : epoch_learning_rate }
                    
                    _, batch_loss = sess.run([train, ce], feed_dict = input_map)
                    batch_acc = accuracy.eval(feed_dict = input_map)

                    train_loss += batch_loss
                    train_acc  += batch_acc
                    train_cnt  += 1
                    
                    list_input_var = []
                    list_label_var = []

            #log
            print('epoch : {}, loss : {}, accuracy : {}'.format(epoch, train_loss / train_cnt, train_acc / train_cnt))

            #save
            saver.save(sess, model_path + model_name.format(epoch))

            if epoch % 5 == 0:

                #test
                list_input_var = []
                list_label_var = []

                _accuracy = 0.0
                _accuracy_sample_cnt = 0

                for _test_data in test_data:
                    test_path, test_label = _test_data

                    frames = []

                    video = cv2.VideoCapture(test_path)
                    while True:
                        ret, frame = video.read()
                        if not ret:
                            break

                        frame = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation = cv2.INTER_CUBIC)
                        frame = np.asarray(frame, dtype = np.float32)
                        frame /= 255.0

                        frames.append(frame)

                        if len(frames) == SEQUENCE:
                            break

                    if len(frames) != SEQUENCE:
                        continue

                    label = one_hot(test_label, CLASSES)

                    list_input_var.append(frames.copy())
                    list_label_var.append(label.copy())

                    if len(list_input_var) == BATCH_SIZE:
                        np_input_var = np.asarray(list_input_var, dtype = np.float32)
                        np_label_var = np.asarray(list_label_var, dtype = np.float32)

                        input_map = { input_var : np_input_var,
                                      label_var : np_label_var }
                
                        batch_acc = accuracy.eval(feed_dict = input_map)
                        
                        _accuracy += batch_acc
                        _accuracy_sample_cnt += 1
                        
                        list_input_var = []
                        list_label_var = []
                
                print('epoch {} test set accuracy :'.format(epoch), _accuracy / _accuracy_sample_cnt)