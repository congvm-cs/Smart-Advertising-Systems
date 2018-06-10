
from keras.preprocessing.image import ImageDataGenerator
import cv2


x = cv2.imread('/home/vmc/Desktop/11.jpg')
x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150

datagen = ImageDataGenerator(
                # rotation_range=10,
                # width_shift_range=0.2,
                # height_shift_range=0.2,
                # rescale=1./255,
                # shear_range=0.2,
                # zoom_range=0.2,
                # horizontal_flip=True,
                fill_mode='nearest',
                brightness_range=[0, 3])

i = 0
for batch in datagen.flow(x, batch_size=1,
                        save_to_dir='/home/vmc/Desktop', save_prefix='cat', save_format='jpeg'):
    i += 1
    if i > 20:
        break  # otherwise the generator would loop indefinitely