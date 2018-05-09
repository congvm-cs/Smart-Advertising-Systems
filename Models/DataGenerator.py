import keras
import numpy as np

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, list_IDs, labels=None, batch_size=32, dim=(32, 32), n_channels=1, 
                n_classes=10, shuffle=True, is_augmented=False):

        'Initialization'
        self.dim = dim                     
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.__on_epoch_end()
        self.is_augmented = is_augmented


    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))


    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)
        return X, y


    def __on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
        
    def __read_image(self, path):
        img = keras.preprocessing.image.load_img(path)
        img = keras.preprocessing.image.img_to_array(img)
        
        if self.is_augmented:
            img = self.__augment_data(img)
        
        img = img/255.0
        img = self.__prewhiten(img)
        return img

  
    def __read_label(self, path):
        label = path.split('/')[-2]
        return label
  
  
    def __prewhiten(self, x):
        mean = np.mean(x)
        std = np.std(x)
        std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
        y = np.multiply(np.subtract(x, mean), 1/std_adj)
        return y 
  

    def __categorize_labels(self, file_path):
        # File name: MFIW_Dataset/train/1/24/024_1_00009890.jpg_0_6250.jpg
        
        file_name = file_path.split('/')[-1]
        
        num_age = file_name.split('_')[0]
        gender = file_name.split('_')[1]

        labels = [0, 0, 0, 0, 0, 0, 0]
        
        age = int(num_age)

        if gender == '1':
            labels[0] = 1   # Female
        else:
            labels[0] = 0   # Male

        if 0 <= age and age <= 10:
            labels[1] = 1
        if 10 < age and age <= 18:
            labels[2] = 1
        elif 18 < age and age <= 25:
            labels[3] = 1
        elif 25 < age and age <= 35:
            labels[4] = 1
        elif 36 < age and age <= 50:
            labels[5] = 1
        elif age > 50:
            labels[6] = 1
        return labels


    def __augment_data(self, image):
        """
        if np.random.random() > 0.5:
            images[i] = random_crop(images[i],4)
        """
        if np.random.random() > 0.75:
            image = keras.preprocessing.image.random_rotation(image, 20, row_axis=0, col_axis=1, channel_axis=2)
        if np.random.random() > 0.75:
            image = keras.preprocessing.image.random_shear(image, 0.2, row_axis=0, col_axis=1, channel_axis=2)
        if np.random.random() > 0.75:
            image = keras.preprocessing.image.random_shift(image, 0.2, 0.2, row_axis=0, col_axis=1, channel_axis=2)
        if np.random.random() > 0.75:
            image = keras.preprocessing.image.random_zoom(image, [0.8, 1.2], row_axis=0, col_axis=1, channel_axis=2)
        if np.random.random() > 0.5:
            image = keras.preprocessing.image.flip_axis(image, axis=1)

        return image


    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, self.n_classes), dtype=int)
    
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = self.__read_image(ID)

            # Store class
            y[i] = __categorize_labels(ID)

        return X, [y[:, 0], y[:, 1::]]