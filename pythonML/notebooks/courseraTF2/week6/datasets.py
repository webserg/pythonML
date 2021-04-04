import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class CiFar10ConvNet(Model):
    def __init__(self):
        super(CiFar10ConvNet, self).__init__()
        self.conv1 = Conv2D(32, 3, activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(10)

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)


def rescale(image, label):
    return image / 255, label


def label_filter(image, label):
    return tf.squeeze(label) != 9


if __name__ == '__main__':
    dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5, 6])
    print(dataset)

    dataset = tf.data.Dataset.from_tensor_slices([[1, 2], [3, 4], [5, 6]])
    print(dataset)

    for elem in dataset:
        print(elem)

    dataset = tf.data.Dataset.from_tensor_slices(tf.random.uniform([128, 5]))
    print(dataset.element_spec)

    dataset = tf.data.Dataset.from_tensor_slices(
        (
            tf.random.uniform([256, 4], minval=1, maxval=10, dtype=tf.int32),
            tf.random.normal([256]))
    )
    print(dataset.element_spec)

    for elem in dataset.take(2):
        print(elem)

    (train_x, train_y), (test_x, test_y) = cifar10.load_data()
    dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    print(dataset.element_spec)

    image_data_gen = ImageDataGenerator(rescale=1 / 255., horizontal_flip=True, height_shift_range=0.2,
                                        fill_mode='nearest', featurewise_center=True)

    dataset = tf.data.Dataset.from_generator(image_data_gen.flow, args=[train_x, train_y],
                                             output_types=(tf.float32, tf.int32),
                                             output_shapes=([32, 32, 32, 3], [32, 1]))
    print(dataset.element_spec)

    dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))

    dataset = dataset.batch(16)
    print(dataset.element_spec)

    dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    dataset.map(rescale)
    dataset.filter(label_filter)
    dataset.shuffle(100)

    dataset = dataset.batch(16, drop_remainder=True)
    print(dataset.element_spec)

    dataset = dataset.repeat(10)  # train for 10 epoch

    model = CiFar10ConvNet()
    history = model.hit(dataset)
