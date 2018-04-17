from AGNet import AGNet
from AGDataset import AGDataset
import argparse
import cv2
import numpy as np
import os

def main(args):
    agNet = AGNet()
    agNet.init()
    
    agdataset = AGDataset()

    [X_train, X_test, y_train, y_test] = agdataset.load_dataset(args)

    print("Shape of X_train: {}".format(X_train.shape))
    print("Shape of X_test: {}".format(X_test.shape))
    print("Shape of y_train: {}".format(y_train.shape))
    print("Shape of y_test: {}".format(y_test.shape))
    
    print("Training ...")
    agNet.train(X_train, y_train, X_test, y_test)

      
def train_on_batch(args):
    print('Load data..')
    train_dir = args.train_dir
    test_dir = args.test_dir
    
    # Init model
    agNet = AGNet()
    agNet.init()
    agdata = AGDataset()

    X_train = []
    X_test = []
    y_train = []
    y_test = []

    train_file_name  = []
    test_file_name = np.array(os.listdir(test_dir))
    

    for subfolder_name in os.listdir(train_dir):
        subfolder_path = os.path.join(train_dir, subfolder_name)

        for subfolder_name1 in os.listdir(subfolder_path):
            subfolder_path1 = os.path.join(subfolder_path, subfolder_name1)
                
            for file_name in os.listdir(subfolder_path1):
                file_path = os.path.join(subfolder_path1, file_name)
                train_file_name.append(file_path)
    # Test phase
    # Load test data

    for subfolder_name in os.listdir(test_dir):
        subfolder_path = os.path.join(train_dir, subfolder_name)

        for subfolder_name1 in os.listdir(subfolder_path):
            subfolder_path1 = os.path.join(subfolder_path, subfolder_name1)
                
            for file_name in os.listdir(subfolder_path1):
                file_path = os.path.join(subfolder_path1, file_name)
                # print(file_path)
                origin_I = cv2.imread(str(file_path))

                # if int(self._IMAGE_DEPTH) == 1:
                origin_I = cv2.cvtColor(origin_I, cv2.COLOR_BGR2GRAY)

                X_test.append(origin_I)
                y_test.append(agdata.categorize_labels(file_name))

    # for i, file_name in enumerate(test_file_name):
    #     file_path = os.path.join(test_dir, file_name)
        
    #     origin_I = cv2.imread(str(file_path))

    #     X_test.append(origin_I)
    #     y_test.append(agdata.categorize_labels(file_name))

    X_test = np.array(X_test)
    X_test = np.reshape(X_test, newshape=(len(X_test), 64, 64, 1))
    
    # Normalize
    X_test = X_test/255.0
    y_test = np.array(y_test)
    

    # Training Phase

    # Split data into every single batch
    num_batches = 10
    range_index = int(len(train_file_name)/num_batches)                             
    train_batches_arr = []

    for epoch in range(100):
        # Shuffle
        np.random.shuffle(train_file_name)

        # Split data into every single batch                         
        train_batches_arr = []

        for i in range(10):                                                       
            if (i + range_index) < len(train_file_name):                                          
                train_batches_arr.append(train_file_name[i*range_index:i*range_index + range_index])
            else:
                train_batches_arr.append(train_file_name[i*range_index:len(train_file_name)])

        print('Epochs: {}'.format(epoch))
        print('-------------------------------------------------------')

        for index, batch in enumerate(train_batches_arr):
            print('--> Batch #{}'.format(index))
            print('-------------------------------------------------------')
            X_train = []
            y_train = []

            for i, file_path in enumerate(batch):
                # file_path = os.path.join(train_dir, file_name)
                origin_I_train = cv2.imread(str(file_path))

                file_name = os.path.split(file_path)
                X_train.append(origin_I_train)
                y_train.append(agdata.categorize_labels(file_name))

            # for subfolder_name in batch:
            #     subfolder_path = os.path.join(train_dir, subfolder_name)

            #     for file_name in os.listdir(subfolder_path):
            #         file_path = os.path.join(subfolder_path, file_name)
            #         # print(file_path)
            #         origin_I = cv2.imread(str(file_path))

            #         # if int(self._IMAGE_DEPTH) == 1:
            #         origin_I = cv2.cvtColor(origin_I, cv2.COLOR_BGR2GRAY)

            #         X_train.append(origin_I)
            #         y_train.append(agdata.categorize_labels(file_name))

            X_train = np.array(X_train)
            X_train = np.reshape(X_train, newshape=(len(X_train), 64, 64, 1))
            # Normalize
            X_train = X_train/255.0
            y_train = np.array(y_train)

            # Training
            agNet.train(X_train, y_train, X_test, y_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', help='train data directory', default=None, type=str)
    parser.add_argument('--test_dir', help='test data directory', default=None, type=str)
    args = parser.parse_args()
    train_on_batch(args)
    # main(args)
