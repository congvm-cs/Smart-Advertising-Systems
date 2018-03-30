from keras.models import Sequential
from keras.layers import Dense, MaxPooling2D, AveragePooling2D, Conv2D, Merge, Flatten
from keras.layers.merge import Concatenate
# Model 1
model1 = Sequential()
model1.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', 
                padding='valid', 
                input_shape=(64, 64, 3)))
model1.add(MaxPooling2D(pool_size=(2, 2)))

model1.add(Conv2D(filters=32, kernel_size=(3, 3), 
                activation='relu', 
                padding='valid'))
model1.add(MaxPooling2D(pool_size=(2, 2)))

model1.add(Flatten())
print(model1.summary())
# Model 2
model2 = Sequential()
model2.add(AveragePooling2D(pool_size=(4, 4), input_shape=(64, 64, 3)))
model2.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', 
                padding='valid'))
model2.add(MaxPooling2D(pool_size=(2, 2)))
model2.add(Flatten())
print(model2.summary())

# Model 3
model3 = Sequential()
model3.add(AveragePooling2D(pool_size=(8, 8), input_shape=(64, 64   , 3)))
model3.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', 
                padding='valid'))
model3.add(MaxPooling2D(pool_size=(2, 2)))
model3.add(Flatten())
print(model3.summary())

merge_model = Merge([model1, model2, model3], mode='concat', concat_axis=-1)

final_model = Sequential()
final_model.add(merge_model)
final_model.add(Dense(512, activation='relu'))
final_model.add(Dense(6, activation='sigmoid'))

print(final_model.summary())