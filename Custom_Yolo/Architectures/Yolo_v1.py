from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Reshape, Dropout, BatchNormalization

tf.keras.backend.clear_session()
yolo = Sequential()

yolo.add(Conv2D(filters=64, kernel_size= (7, 7), strides=(1, 1), input_shape =(448, 448, 3), padding = 'same', kernel_regularizer=l2(5e-4)))
yolo.add(BatchNormalization())
yolo.add(LeakyReLU(alpha=0.1))
yolo.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding = 'same'))

yolo.add(Conv2D(filters=192, kernel_size= (3, 3), padding = 'same', kernel_regularizer=l2(5e-4)))
yolo.add(BatchNormalization())
yolo.add(LeakyReLU(alpha=0.1))
yolo.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding = 'same'))

yolo.add(Conv2D(filters=128, kernel_size= (1, 1), padding = 'same', kernel_regularizer=l2(5e-4)))
yolo.add(BatchNormalization())
yolo.add(LeakyReLU(alpha=0.1))
yolo.add(Conv2D(filters=256, kernel_size= (3, 3), padding = 'same', kernel_regularizer=l2(5e-4)))
yolo.add(BatchNormalization())
yolo.add(LeakyReLU(alpha=0.1))


yolo.add(Conv2D(filters=256, kernel_size= (1, 1), padding = 'same', kernel_regularizer=l2(5e-4)))
yolo.add(BatchNormalization())
yolo.add(LeakyReLU(alpha=0.1))
yolo.add(Conv2D(filters=512, kernel_size= (3, 3), padding = 'same', kernel_regularizer=l2(5e-4)))
yolo.add(BatchNormalization())
yolo.add(LeakyReLU(alpha=0.1))


yolo.add(Conv2D(filters=512, kernel_size= (1, 1), padding = 'same', kernel_regularizer=l2(5e-4)))
yolo.add(BatchNormalization())
yolo.add(LeakyReLU(alpha=0.1))
yolo.add(Conv2D(filters=1024, kernel_size= (3, 3), padding = 'same', kernel_regularizer=l2(5e-4)))
yolo.add(BatchNormalization())
yolo.add(LeakyReLU(alpha=0.1))
yolo.add(Conv2D(filters=512, kernel_size= (1, 1), padding = 'same', kernel_regularizer=l2(5e-4)))
yolo.add(BatchNormalization())
yolo.add(LeakyReLU(alpha=0.1))
yolo.add(Conv2D(filters=1024, kernel_size= (3, 3), padding = 'same', kernel_regularizer=l2(5e-4)))
yolo.add(BatchNormalization())
yolo.add(LeakyReLU(alpha=0.1))

yolo.add(Reshape((200704,), input_shape=(14,14, 1024)))

yolo.add(Dense(245, activation="softmax"))

yolo.add(Reshape((7,7,5), input_shape=(245,)))
