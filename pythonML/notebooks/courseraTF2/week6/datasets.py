import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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
