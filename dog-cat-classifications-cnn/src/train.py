import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import RMSprop
import config

def generator():
    train_datgen = ImageDataGenerator(rescale=1./255)

    train_genrator = train_datgen.flow_from_directory(
        config.TRAIN_DIRECTORY,
        target_size=(256, 256),
        color_mode='rgb',
        batch_size=100,
        shuffle=True,
        class_mode='binary'
    )

    validation_genrator = train_datgen.flow_from_directory(
        config.VALIDATION_DIRECTORY,
        target_size=(256, 256),
        color_mode='rgb',
        batch_size=50,
        shuffle=True,
        class_mode='binary'
    )
    return train_genrator,validation_genrator

def model():
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Conv2D(16,(3,3),activation='relu',input_shape=(256,256,3)),
            tf.keras.layers.MaxPool2D(2,2),
            tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
            tf.keras.layers.MaxPool2D(2, 2),
            tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
            tf.keras.layers.MaxPool2D(2, 2),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.MaxPool2D(2, 2),
            tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
            tf.keras.layers.MaxPool2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512,activation='relu'),
            tf.keras.layers.Dense(1,activation='sigmoid')
        ]
    )
    return model

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('acc')>0.97):
            print("loss is low cancelling")
            self.model.stop_training = True

if __name__ == '__main__':
    callbacks = myCallback()
    train_generator,validation_generator = generator()
    model_used = model()
    model_used.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['acc'])
    model_used.fit_generator(
        train_generator,
        steps_per_epoch=115,
        epochs=20,
        validation_data=validation_generator,
        validation_steps=20,
        callbacks=[callbacks]
    )
    model_used.save_weights(f'{config.MODEL_OUTPUT}/model.h5')
    print(model_used.summary())