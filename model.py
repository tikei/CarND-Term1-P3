import csv
import cv2
import numpy as np
import tensorflow as tf
import sklearn

DATA_SPLIT = 0.2 # validation set size
DIR_NAME = './data_4/'

# Flags
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('pre_trained', '', "Pre-trained model (.h5)")

# Hyperparameters
EPOCHS = 5 
LEARN_RATE = 0.001
AUGMENT = True # augment the data
CORRECTION = 0.25 # steering correction for left and right camera


def samples_generator(samples, batch_size=32, augment=False):
    num_samples = len(samples)

    while True:
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            
            dir_name = DIR_NAME + 'IMG/'
            images = []
            angles = []
            for batch_sample in batch_samples:
                center_name = dir_name + batch_sample[0].split('/')[-1]
                center_image = cv2.imread(center_name)[:,:,::-1]
                center_angle = float(batch_sample[3])
                left_name = dir_name + batch_sample[1].split('/')[-1]
                left_image = cv2.imread(left_name)[:,:,::-1]
                left_angle = center_angle + CORRECTION
                right_name = dir_name + batch_sample[2].split('/')[-1]
                right_image = cv2.imread(right_name)[:,:,::-1]
                right_angle = center_angle - CORRECTION

                images.append(left_image)
                angles.append(left_angle)
                images.append(right_image)
                angles.append(right_angle)

                if abs(center_angle) > 0.7:
                    images.append(center_image)
                    angles.append(center_angle)
                    if augment:
                        images.append(np.fliplr(center_image))
                        angles.append(-center_angle)

			

            X_train = np.array(images)
            y_train = np.array(angles)

            yield sklearn.utils.shuffle(X_train, y_train)


def main(_):

    samples = []
    with open(DIR_NAME + 'driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        _ = next(reader, None) # ignore headers
        for line in reader:
            samples.append(line)

    from sklearn.model_selection import train_test_split
    train_samples, validation_samples = train_test_split(samples, test_size=DATA_SPLIT)
    print('TRAIN SAMPLES: {}'.format(len(train_samples)))


    train_generator = samples_generator(train_samples, batch_size = 32, augment=AUGMENT)
    validation_generator = samples_generator(validation_samples, batch_size = 32, augment=AUGMENT)


    if FLAGS.pre_trained:
        print('FILE pre_trained passed!')
        from keras.models import load_model
        model = load_model(FLAGS.pre_trained)

    else:
        # Network architecture as per Nvidia paper
        from keras.models import Sequential
        from keras.layers.core import Flatten, Dense, Lambda
        from keras.layers.convolutional import Cropping2D, Convolution2D
        from keras.optimizers import Adam

        model = Sequential()
        model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape = (160, 320, 3)))
        model.add(Cropping2D(cropping=((60, 25), (0, 0))))
        model.add(Convolution2D(24, 5, 5, border_mode='same', subsample=(2, 2)))
        model.add(Convolution2D(36, 5, 5, border_mode='same', subsample=(2, 2)))
        model.add(Convolution2D(48, 5, 5, border_mode='same', subsample=(2, 2)))
        model.add(Convolution2D(64, 3, 3, border_mode='same'))
        model.add(Convolution2D(64, 3, 3, border_mode='same'))
        model.add(Flatten())
        model.add(Dense(100))
        model.add(Dense(50))
        model.add(Dense(10))
        model.add(Dense(1))

        adam = ADAM(lr=LEARN_RATE)
        model.compile(loss='mse', optimizer=adam)


    if AUGMENT:
        samp_per_ep = len(train_samples) * 4
        n_val_samp = len(validation_samples) * 4
    else:
        samp_per_ep = len(train_samples) 
        n_val_samp = len(validation_samples)

    model.fit_generator(train_generator, samples_per_epoch=samp_per_ep, 
            validation_data=validation_generator, 
            nb_val_samples=n_val_samp, nb_epoch=EPOCHS, verbose=1)

    from time import strftime, gmtime
    file_to_save = 'model-' + strftime("%Y-%m-%d_%H%M%S", gmtime()) + '.h5'
    model.save(file_to_save)

if __name__ == '__main__':
    tf.app.run()
