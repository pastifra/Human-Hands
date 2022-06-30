tf.keras.backend.clear_session()
yolo = Sequential()

yolo.add(tf.keras.layers.Conv2D(64, (7, 7), padding="same", strides = (2,2), input_shape=(448,448,3)))
yolo.add(LeakyReLU(alpha=0.1))
yolo.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2),strides = (2,2)))

yolo.add(tf.keras.layers.Conv2D(192, (3, 3), padding="same", strides = (2,2)))
yolo.add(LeakyReLU(alpha=0.1))
yolo.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2),strides = (2,2)))

yolo.add(tf.keras.layers.Conv2D(128, (1, 1), padding="same"))
yolo.add(LeakyReLU(alpha=0.1))
yolo.add(tf.keras.layers.Conv2D(256, (3, 3), padding="same"))
yolo.add(LeakyReLU(alpha=0.1))
yolo.add(tf.keras.layers.Conv2D(256, (1, 1), padding="same"))
yolo.add(LeakyReLU(alpha=0.1))
yolo.add(tf.keras.layers.Conv2D(512, (3, 3), padding="same"))
yolo.add(LeakyReLU(alpha=0.1))
yolo.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2),strides = (2,2)))

yolo.add(tf.keras.layers.Conv2D(256, (1, 1), padding="same"))
yolo.add(LeakyReLU(alpha=0.1))
yolo.add(tf.keras.layers.Conv2D(512, (3, 3), padding="same"))
yolo.add(LeakyReLU(alpha=0.1))
yolo.add(tf.keras.layers.Conv2D(256, (1, 1), padding="same"))
yolo.add(LeakyReLU(alpha=0.1))
yolo.add(tf.keras.layers.Conv2D(512, (3, 3), padding="same"))
yolo.add(LeakyReLU(alpha=0.1))
yolo.add(tf.keras.layers.Conv2D(256, (1, 1), padding="same"))
yolo.add(LeakyReLU(alpha=0.1))
yolo.add(tf.keras.layers.Conv2D(512, (3, 3), padding="same"))
yolo.add(LeakyReLU(alpha=0.1))
yolo.add(tf.keras.layers.Conv2D(256, (1, 1), padding="same"))
yolo.add(LeakyReLU(alpha=0.1))
yolo.add(tf.keras.layers.Conv2D(512, (3, 3), padding="same"))
yolo.add(LeakyReLU(alpha=0.1))
yolo.add(tf.keras.layers.Conv2D(512, (1, 1), padding="same"))
yolo.add(LeakyReLU(alpha=0.1))
yolo.add(tf.keras.layers.Conv2D(1024, (3, 3), padding="same"))
yolo.add(LeakyReLU(alpha=0.1))
yolo.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2),strides = (2,2)))

yolo.add(tf.keras.layers.Conv2D(512, (1, 1), padding="same"))
yolo.add(LeakyReLU(alpha=0.1))
yolo.add(tf.keras.layers.Conv2D(1024, (3, 3), padding="same"))
yolo.add(LeakyReLU(alpha=0.1))
yolo.add(tf.keras.layers.Conv2D(512, (1, 1), padding="same"))
yolo.add(LeakyReLU(alpha=0.1))

#COMMENTED LAYERS ARE FOR YOLO OBJECT DETECTION MODEL

#yolo.add(tf.keras.layers.Conv2D(1024, (3, 3), padding="same"))
#yolo.add(LeakyReLU(alpha=0.1))
#yolo.add(tf.keras.layers.Conv2D(1024, (3, 3), padding="same", strides = (2,2)))
#yolo.add(LeakyReLU(alpha=0.1))

#yolo.add(tf.keras.layers.Conv2D(1024, (3, 3), padding="same"))
#yolo.add(LeakyReLU(alpha=0.1))
#yolo.add(tf.keras.layers.Conv2D(1024, (3, 3), padding="same"))
#yolo.add(LeakyReLU(alpha=0.1))

#yolo.add(tf.keras.layers.Reshape((16384,), input_shape=(3,3,1024)))

#yolo.add(Dense(4096))
#yolo.add(LeakyReLU(alpha=0.1))

#yolo.add(Dense(245, activation = 'relu'))

#yolo.add(tf.keras.layers.Reshape((7,7,5), input_shape=(245,)))

#FOLLOWING LAYERS ARE FOR YOLO CLASSIFICATION PRE-TRAINING MODEL
yolo.add(tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2,2)))
yolo.add(tf.keras.layers.Reshape((4608,), input_shape=(3,3,512)))
yolo.add(Dense(2))
yolo.add(Activation('softmax'))

yolo.summary()