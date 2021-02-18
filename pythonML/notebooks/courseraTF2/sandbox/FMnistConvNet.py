import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.layers import Softmax
from tensorflow.keras.models import Sequential

if __name__ == '__main__':

    fashion_mnist_data = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist_data.load_data()

    labels = [
        'T-shirt/top',
        'Trouser',
        'Pullover',
        'Dress',
        'Coat',
        'Sandal',
        'Shirt',
        'Sneaker',
        'Bag',
        'Ankle boot'
    ]

    model = Sequential([
        Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((3, 3)),
        Flatten(),
        Dense(10),
        Softmax()])
    model.summary()

    opt = tf.keras.optimizers.Adam(learning_rate=0.005)
    acc = tf.keras.metrics.SparseCategoricalAccuracy()
    mae = tf.keras.metrics.MeanAbsoluteError()
    model.compile(optimizer=opt,
                  loss='sparse_categorical_crossentropy',
                  metrics=[acc, mae]
                  )

    print(model.loss)
    print(model.optimizer)
    print(model.metrics)

    # Rescale the image values so that they lie in between 0 and 1.

    train_images = train_images / 255.
    test_images = test_images / 255.

    # Display one of the images
    i = 0
    img = train_images[i, :, :]
    plt.imshow(img)
    plt.show()
    print(f"label: {labels[train_labels[i]]}")

    history = model.fit(train_images[..., np.newaxis], train_labels, epochs=4, batch_size=256)

    df = pd.DataFrame(history.history)
    df.head()
    # Make a plot for the loss
    loss_plot = df.plot(y="loss", title="Loss vs Epochs", legend=False)
    loss_plot.set(xlabel="Epochs", ylabel="Loss")

    test_loss, test_accurace, test_mae = model.evaluate(test_images[..., np.newaxis], test_labels)

    random_inx = np.random.choice(test_images.shape[0])

    test_image = test_images[random_inx]
    plt.imshow(test_image)
    plt.show()
    print(f"Label: {labels[test_labels[random_inx]]}")

    predictions = model.predict(test_image[np.newaxis, ..., np.newaxis])
    print(labels[np.argmax(predictions)])
