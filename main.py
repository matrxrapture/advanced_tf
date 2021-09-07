import tensorflow as tf
from tensorflow.keras.utils import plot_model

import pydot
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Input


# sequencial
def build_model_with_sequential():
    seq_model = tf.keras.models.Sequential(Flatten(28.28)), Dense(
        128, activation='relu'), Dense(10, activation='softmax')
    return seq_model

# Funcional


def build_model_with_functional():
    input_layer = tf.keras.Input(shape=(28, 28))
    flatten_layer = Flatten()(input_layer)
    dense_layer = Dense(128, activation='relu')(flatten_layer)
    predition = Dense(10, activation='softmax')(dense_layer)
    func_model = Model(input_layer, predition)
    return func_model


def visualize(model):
    plot_model(model, show_shapes=True,
               show_layer_names=True, to_file='model.png')


def train(model):
    mnist = tf.keras.datasets.fashion_mnist
    (training_images, training_labels), (test_images, test_labels) = mnist.load_data()
    training_images = training_images / 255
    test_images = test_images/255

    # configure ,train and evaluate model
    model.compile(optimizer=tf.optimizers.Adam(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(training_images, training_labels, epochs=50)
    model.evaluate(test_images, test_labels)


if __name__ == "__main__":
    model2 = build_model_with_functional()
    plot_model(model2, show_shapes=True,
               show_layer_names=True, to_file='model.png')
    print("hello")
    print(model2)
    train(model2)
