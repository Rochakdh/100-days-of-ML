import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import config

def generator():
    train_datgen = ImageDataGenerator(rescale=1.0/255)

    train_genrator = train_datgen.flow_from_directory(
        config.TRAIN_DIRECTORY,
        target_size=(256, 256),
        color_mode='rgb',
        batch_size=128,
    )

    validation_genrator = train_datgen.flow_from_directory(
        config.VALIDATION_DIRECTORY,
        target_size=(256, 256),
        color_mode='rgb',
        batch_size=32,
    )

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('acc')>0.97):
            print("loss is low cancelling")
            self.model.stop_training = True

if __name__ == '__main__':
    generator()