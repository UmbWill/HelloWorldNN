from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

from keras.preprocessing.image import ImageDataGenerator

from IPython.display import display
from PIL import Image

classifier = Sequential()
classifier.add(Convolution2D(32,3,3, input_shape = (100,100,3), activation = "relu"))
classifier.add(MaxPooling2D(pool_size = (2,2)))
classifier.add(Flatten())
classifier.add(Dense(output_dim = 128, activation = "relu"))
classifier.add(Dense(output_dim = 1, activation = "sigmoid"))
classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

train_datagen = ImageDataGenerator(rescale = 1.255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory("dataset/Fruit-Images-Dataset-master/Training", target_size=(100,100), batch_size = 32, class_mode = "binary")
test_set = test_datagen.flow_from_directory("dataset/Fruit-Images-Dataset-master/Test", target_size=(100,100), batch_size = 32, class_mode = "binary")

classifier.fit_generator(training_set, steps_per_epoch=8000,epochs=10,validation_data = test_set, validation_steps = 800)

