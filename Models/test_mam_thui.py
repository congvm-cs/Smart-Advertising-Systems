import numpy as np
# import tensorflow as tf


def multi_labels_accuracy(y_true, y_pred):
    acc = K.equal(y_true, y_pred)
    return K.mean(K.all(acc, axis=-1) == True)
    # return acc

def _multi_labels_accuracy(y_true, y_pred):
    acc = np.equal(y_true, np.round(y_pred))
    acc = np.all(acc, axis=-1)
    # acc = tf.cast(acc, tf.float32
    return np.mean(acc)

y = [[1, 0, 1, 0], [0, 0, 1, 1], [0, 0, 1, 1]]
pred = [[1, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]]

print(_multi_labels_accuracy(y, pred))

x = [[1, 1, 1, 1], 
    [1, 1, 1, 1]]

# print(K.all(x, axis=-1) == 1)

# with tf.Session() as sess:
#     print(sess.run(_multi_labels_accuracy(y, pred)))
