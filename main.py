from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, Input
from keras import backend as K
from keras import applications, optimizers
from keras.utils import plot_model
import numpy as np
import os
import time

"""
In our setup, we:
- created a data/ folder
- created train/ and validation/ subfolders inside data/
- created cats/ and dogs/ subfolders inside train/ and validation/
We have 1000 training examples for each class, and 400 validation examples for each class
"""

# For demonstration
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')


# --------------------------------------- Model Parameters -------------------------------------------------------------
# target dimensions of our images.
img_width, img_height = 150, 150

train_data_dir = 'data/train'
validation_data_dir = 'data/validation'

nb_train_samples = 2000
nb_validation_samples = 800
epochs = 50 # try other values and probe changes in loss values
batch_size = 16 # Other values to try: 8, 32 or 64

save_model_train_from_scratch_path = 'first_try.h5'
top_model_weights_path = 'bottleneck_fc_model.h5'
save_bottleneck_features_train_path = 'bottleneck_features_train.npy'
save_bottleneck_features_validation_path = 'bottleneck_features_validation.npy'
# -----------------------------------------------xxx--------------------------------------------------------------------

def preview_data_augmentation():
    img = load_img('data/train/cats/cat.173.jpg')
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)

    i = 0
    for batch in datagen.flow(x, batch_size=1, save_to_dir='preview', save_prefix='cat', save_format='jpeg'):
        i+=1
        if i > 20:
            break

def train_from_scratch():

    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)

    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    plot_model(model, to_file='model.png')
    print(model.summary())

    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
        rescale = 1. / 255,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True)

    # this is the augmentation configuration we will use for testing:
    # only rescaling
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    # this is a generator that will read pictures found in
    # sub-folders of 'data/train', and indefinitely generate
    # batches of augmented image data
    train_generator = train_datagen.flow_from_directory(
            'data/train',  # this is the target directory
            target_size=(img_width, img_height),  # all images will be resized to 150x150
            batch_size=batch_size,
            class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels

    # this is a similar generator, for validation data
    validation_generator = test_datagen.flow_from_directory(
            'data/validation',
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode='binary')

    # Training the model using fit_generator method
    model.fit_generator(
            train_generator,
            steps_per_epoch=nb_train_samples // batch_size,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=nb_validation_samples // batch_size)

    model.save_weights(save_model_train_from_scratch_path)

def save_bottleneck_features():

    """
    To save computation time on VGG16, we save the computed features of last conv layer for all samples
    """
    datagen = ImageDataGenerator(rescale=1. / 255)

    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')

    if not os.path.exists(save_bottleneck_features_train_path):
        generator = datagen.flow_from_directory(
            train_data_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode=None,
            shuffle=False)

        bottleneck_features_train = model.predict_generator(generator, nb_train_samples // batch_size)

        np.save(save_bottleneck_features_train_path, bottleneck_features_train)
    else:
        print("Features for training data already exist at {}".format(save_bottleneck_features_train_path))

    if not os.path.exists(save_bottleneck_features_validation_path):
        generator = datagen.flow_from_directory(
            validation_data_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode=None,
            shuffle=False)

        bottleneck_features_validation = model.predict_generator(generator, nb_validation_samples // batch_size)

        np.save(save_bottleneck_features_validation_path, bottleneck_features_validation)
    else:
        print("Features for validation data already exist at {}".format(save_bottleneck_features_validation_path))

def train_top_model():

    # For Computational efficiency. Running VGG16 is expensive, so we want to only do it once.
    save_bottleneck_features()

    train_data = np.load(save_bottleneck_features_train_path)
    train_labels = np.array([0] * (nb_train_samples // 2) + [1] * (nb_train_samples // 2))

    validation_data = np.load(save_bottleneck_features_validation_path)
    validation_labels = np.array([0] * (nb_validation_samples // 2) + [1] * (nb_validation_samples // 2))

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels))

    model.save_weights(top_model_weights_path)

def fine_tune():

    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)

    # build the VGG16 network
    model_vgg16_conv = applications.VGG16(include_top=False, weights='imagenet', input_shape=input_shape)

    print('Model loaded.')

    # build a classifier model to put on top of the convolutional model
    input_FC = Input(shape=model_vgg16_conv.output_shape[1:])
    x = Flatten()(input_FC)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1, activation='sigmoid')(x)
    FC_model = Model(input=input_FC, output=x)

    # Note that it is necessary to start with a fully-trained
    # classifier, including the top classifier,
    # in order to successfully do fine-tuning
    FC_model.load_weights(top_model_weights_path)

    # add the model on top of the convolutional base
    input_model = Input(shape=input_shape)
    output_vgg16_conv = model_vgg16_conv(input_model)
    prediction = FC_model(output_vgg16_conv)
    model = Model(input=input_model, output=prediction)

    # set the first 25 layers (up to the last conv block)
    # to non-trainable (weights will not be updated)
    for layer in model.layers[:25]:
        layer.trainable = False

    # compile the model with a SGD/momentum optimizer
    # and a very slow learning rate.
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                  metrics=['accuracy'])

    # prepare data augmentation configuration
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary')

    # fine-tune the model
    model.fit_generator(
        train_generator,
        samples_per_epoch=nb_train_samples,
        epochs=epochs,
        validation_data=validation_generator,
        nb_val_samples=nb_validation_samples)


if __name__ == '__main__':

    start = time.time()
    # preview_data_augmentation()

    # # Method-1: Train a CNN from scratch
    train_from_scratch()

    # # Method-2: Freeze the Convolutional layers and learn only the FC Layers
    train_top_model()

    # # Method-3: Fine tuning
    # fine_tune()

    print("Time taken: {}".format(round(time.time()-start, 2)))



