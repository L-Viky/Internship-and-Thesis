import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers, constraints

dataset, info = tfds.load("speech_commands", data_dir = "/media/viky/Linux Dati1/Internship-and-Thesis", with_info=True, as_supervised=True)

class BinaryConstraint(constraints.Constraint):
    def __call__(self, w):
        # Apply binary constraint to weights
        return tf.where(tf.equal(w, 0.0), tf.ones_like(w), tf.sign(w)) #return 1 when values positive and when values zero and -1 when values negative
    
def BinaryActivation (x):
    #take the sign
    y = tf.sign(x)
    # check for 0 and replate it with 1
    y = tf.where(tf.equal(y, 0), tf.one_like(y), y)

    return y


def BinaryConv2D(filters, kernel_size, strides=(1, 1), padding='valid', **kwargs):
    # Create a Conv2D layer with the binary constraint applied to its kernel
    conv = layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        kernel_constraint=BinaryConstraint(),  # Apply the binary constraint here
        **kwargs
    )
    return conv


def make_model():
    '''
    This function creates a convolutional neural network model for Keyword Spotting.
    Our CNN has the following architecture:
        - Full Precision Convolutional Layer 1 : (51,32,64)
        - Binary Convolutional Layer 2 : (51,32,128)
        - Binary Convolutional Layer 3 : (26,16,128)
        - Binary Convolutional Layer 4 : (26,16,192)
        - Binary Convolutional Layer 5 : (26,16,192)
        - Full Precision Convolutional Layer 6 : (26,16,12)
        - Average Polling Layer 7 : (,12)
    '''
    
    # Input shape for spectrograms (assuming 51x32 input)
    inputs = keras.Input(shape=(51, 32, 1))
    
    # Layer 1: Full Precision Convolutional Layer (51,32,64)
    x1 = layers.Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu')(inputs)
    x1 = layers.BatchNormalization()(x1)
    
    # Layer 2: Binary Convolutional Layer (51,32,128)
    x2 = BinaryConv2D(128, kernel_size=(3, 3), padding='same', activation=BinaryActivation)(x1)
    x2 = layers.BatchNormalization()(x2)
    
    # Layer 3: Binary Convolutional Layer (26,16,128) - requires downsampling
    x3 = BinaryConv2D(128, kernel_size=(3, 3), strides=(2, 2), padding='same', activation=BinaryActivation)(x2)
    x3 = layers.BatchNormalization()(x3)
    
    # Layer 4: Binary Convolutional Layer (26,16,192)
    x4 = BinaryConv2D(192, kernel_size=(3, 3), padding='same', activation=BinaryActivation)(x3)
    x4 = layers.BatchNormalization()(x4)
    
    # Layer 5: Binary Convolutional Layer (26,16,192)
    x5 = BinaryConv2D(192, kernel_size=(3, 3), padding='same', activation=BinaryActivation)(x4)
    x5 = layers.BatchNormalization()(x5)
    
    # Layer 6: Full Precision Convolutional Layer (26,16,12)
    x6 = layers.Conv2D(12, kernel_size=(1, 1), padding='same', activation='relu')(x5)
    x6 = layers.BatchNormalization()(x6)
    
    # Layer 7: Average Pooling Layer (,12) - global average pooling
    x7 = layers.GlobalAveragePooling2D()(x6)
    
    # Output layer with softmax for classification
    outputs = layers.Activation('softmax')(x7)

    return keras.Model(inputs=inputs, outputs=outputs)

model = make_model()

model.summary()
