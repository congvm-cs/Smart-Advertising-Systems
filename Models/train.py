from FGNet import FGNet
from FGDataset import FGDataset
import argparse

def main(args):
    fgnet = FGNet()
    fgdataset = FGDataset()

    [X_train, X_test, y_train, y_test] = fgdataset.load_dataset(args)

    print("Shape of X_train: {}".format(X_train.shape))
    print("Shape of X_test: {}".format(X_test.shape))
    print("Shape of y_train: {}".format(y_train.shape))
    print("Shape of y_test: {}".format(y_test.shape))
    
    print("Training ...")
    fgnet.train(X_train, X_test, y_train, y_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', help='train data directory', default=None, type=str)
    parser.add_argument('--test_dir', help='test data directory', default=None, type=str)
    args = parser.parse_args()
    main(args)
