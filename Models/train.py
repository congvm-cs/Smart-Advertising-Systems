from AGNet import AGNet
from AGDataset import AGDataset
import argparse
import cv2
import numpy as np
import os

def main(args):
    agNet = AGNet()
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
        batch_size = args.batch_size
        
        agNet = AGNet()

        X_train = []
        X_test = []
        y_train = []
        y_test = []

        train_file_name  = os.listdir(train_dir)
        test_file_name = os.listdir(test_dir)
        
        # Test phase
        # Load test data
        for i, file_name in enumerate(test_file_name):
            file_path = os.path.join(test_dir, file_name)
            
            origin_I = cv2.imread(str(file_path))

            if int(self._IMAGE_DEPTH) == 1:
                origin_I = cv2.cvtColor(origin_I, cv2.COLOR_BGR2GRAY)

            X_test.append(origin_I)
            y_test.append(self.categorize_labels(file_name))

        X_test = np.array(X_test)
        X_test = np.reshape(X_test, newshape=(len(X_test), self._IMAGE_SIZE, self._IMAGE_SIZE, self._IMAGE_DEPTH))
        X_test = X_test/255.0
        y_test = np.array(y_test)
        

        # Shuffle
        train_file_name = np.random.shuffle(train_file_name)
        
        # Split data into every single batch
        train_batches_arr = np.split(train_file_name, 10)
        
        for _ in range(100):
            for batch in train_batches_arr:
                for i, file_name in enumerate(batch):
                    file_path = os.path.join(train_dir, file_name)
                    
                    origin_I_train = cv2.imread(str(file_path))

                    if int(self._IMAGE_DEPTH) == 1:
                        origin_I_train = cv2.cvtColor(origin_I_train, cv2.COLOR_BGR2GRAY)

                    X_train.append(origin_I_train)
                    y_train.append(self.categorize_labels(file_name))

                X_train = np.array(X_train)
                X_train = np.reshape(X_train, newshape=(len(X_train), self._IMAGE_SIZE, self._IMAGE_SIZE, self._IMAGE_DEPTH))
                # Normalize
                X_train = X_train/255.0
                y_train = np.array(y_train)
                agNet.train(X_train, y_train, X_test, y_test)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', help='train data directory', default=None, type=str)
    parser.add_argument('--test_dir', help='test data directory', default=None, type=str)
    args = parser.parse_args()
    train_on_batch(args)
