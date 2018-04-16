import numpy as K

def multi_labels_accuracy(y_true, y_pred):
    acc = K.equal(y_true, y_pred)
    return K.mean(K.all(acc, axis=-1) == True)
    # return acc

y = [[1, 0, 1, 0], [0, 0, 1, 1], [0, 0, 1, 1]]
pred = [[1, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]]

print(multi_labels_accuracy(y, pred))

x = [[1, 1, 1, 1], 
    [1, 1, 1, 1]]

# print(K.all(x, axis=-1) == 1)