
tf.keras.backend.clear_session()

length =  448
width = 448
input_shape = (length, width, 3)

yolo = Sequential()

yolo.add(Conv2D(32, (3, 3), input_shape=input_shape, activation='relu', padding='same'))
yolo.add(MaxPooling2D((2,2)))
yolo.add(Dropout(0.1))

yolo.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
yolo.add(Conv2D(32, (1, 1), activation='relu', padding='same'))
yolo.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
yolo.add(MaxPooling2D((2,2)))
yolo.add(Dropout(1e-3))

yolo.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
yolo.add(Conv2D(64, (1, 1), activation='relu', padding='same'))
yolo.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
yolo.add(MaxPooling2D((2,2)))
yolo.add(Dropout(1e-3))

yolo.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
yolo.add(Conv2D(128, (1, 1), activation='relu', padding='same'))
yolo.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
yolo.add(MaxPooling2D((2,2)))
yolo.add(Dropout(1e-3))

yolo.add(Reshape((200704,), input_shape=(28,28, 256)))
yolo.add(Dense(245, activation="softmax"))

yolo.add(Reshape((7,7,5), input_shape=(245,)))

print (yolo.summary())
