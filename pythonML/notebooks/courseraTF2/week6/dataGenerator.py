from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D

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

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    image_data_gen = ImageDataGenerator(rescale=1 / 255., horizontal_flip=True, height_shift_range=0.2,
                                        fill_mode='nearest', featurewise_center=True)

    image_data_gen.fit(x_train)

    train_dataget = image_data_gen.flow(x_train, y_train, batch_size=16)

    model = CiFar10ConvNet()
    model.fit_generator(train_dataget, epochs=20)

