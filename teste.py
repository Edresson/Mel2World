import tensorflow as tf
import numpy as np

myname = "mels/LJ018-0285.npy"   
print(np.load(myname))

def printsomthing(name):
    print(name)
    return np.load(name)

op = tf.py_func(printsomthing,[myname],[tf.float32])
session = tf.Session()
print(session.run(op))
print(tf.__version__)
