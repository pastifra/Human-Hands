#--------------#
#| YOLO SMALL |#
#--------------#

tf.keras.backend.clear_session()

yolo = Sequential()

yolo.add(tf.keras.layers.Conv2D(32, (7, 7), padding="same", strides = (1,1), input_shape=(416,416,3), kernel_regularizer=l2(5e-4)))
yolo.add(tf.keras.layers.BatchNormalization()) #Batch normalization needs to be executed before lrelu apparently 
yolo.add(LeakyReLU(alpha=0.1))
yolo.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2),strides = (2,2), padding = 'same'))

yolo.add(tf.keras.layers.Conv2D(64, (3, 3), padding="same", strides = (2,2), kernel_regularizer=l2(5e-4)))
yolo.add(tf.keras.layers.BatchNormalization()) #Batch normalization needs to be executed before lrelu apparently 
yolo.add(LeakyReLU(alpha=0.1))
yolo.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2),strides = (2,2), padding = 'same'))

yolo.add(tf.keras.layers.Conv2D(64, (1, 1), padding="same", strides = (2,2), kernel_regularizer=l2(5e-4)))
yolo.add(tf.keras.layers.BatchNormalization()) #Batch normalization needs to be executed before lrelu apparently 
yolo.add(LeakyReLU(alpha=0.1))
yolo.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2),strides = (2,2), padding = 'same'))

yolo.add(tf.keras.layers.Conv2D(128, (3, 3), padding="same", strides = (2,2), kernel_regularizer=l2(5e-4)))
yolo.add(tf.keras.layers.BatchNormalization()) #Batch normalization needs to be executed before lrelu apparently 
yolo.add(LeakyReLU(alpha=0.1))
yolo.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2),strides = (2,2), padding = 'same'))


yolo.add(tf.keras.layers.Reshape((2048,), input_shape=(4,4,128)))


yolo.add(Dense(320))
yolo.add(tf.keras.layers.Reshape((8,8,5), input_shape=(245,)))

yolo.summary()