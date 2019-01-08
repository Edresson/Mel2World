 # -*- coding: utf-8 -*-
# /usr/bin/python3

from __future__ import print_function

from tqdm import tqdm

from data_load import get_batch
from hyperparams import Hyperparams as hp
from modules import *
import tensorflow as tf
from utils import *

def Mel2World(Y, training=True):
    '''
    Args:
      Y: Melspectrogram Predictions. (B, T/r, n_mels)

    Returns:
      W: Concatenated  World Vocoder Features. (B, 10*T/r, 3+n_fft/2)
    '''

    i = 1 # number of layers

    # -> (B, T/r, c) 
    tensor = conv1d(Y,
                    filters=hp.c,
                    size=1,
                    rate=1,
                    dropout_rate=hp.dropout_rate,
                    training=training,
                    scope="C_{}".format(i)); i += 1
    for j in range(2):
        tensor = hc(tensor,
                      size=3,
                      rate=3**j,
                      dropout_rate=hp.dropout_rate,
                      training=training,
                      scope="HC_{}".format(i)); i += 1
    
    # -> (B, T/r, c) -> (B, 2*T/r, c)
    tensor = conv1d_transpose(tensor,
                                scope="D_{}".format(i),
                                dropout_rate=hp.dropout_rate,
                                training=training,); i += 1
    for j in range(2):
            tensor = hc(tensor,
                            size=3,
                            rate=3**j,
                            dropout_rate=hp.dropout_rate,
                            training=training,
                            scope="HC_{}".format(i)); i += 1
    # -> (B, 2*T/r, c) -> (B, 2*4*T/r, c)
    tensor = conv1d_transpose(tensor,
                                stride=4,
                                scope="D_{}".format(i),
                                dropout_rate=hp.dropout_rate,
                                training=training,); i += 1
    for j in range(2):
            tensor = hc(tensor,
                            size=3,
                            rate=3**j,
                            dropout_rate=hp.dropout_rate,
                            training=training,
                            scope="HC_{}".format(i)); i += 1
            
    # -> (B, 10*T/r, 2*c)
    tensor = conv1d(tensor,
                    filters=2*hp.c,
                    size=1,
                    rate=1,
                    dropout_rate=hp.dropout_rate,
                    training=training,
                    scope="C_{}".format(i)); i += 1
    for _ in range(2):
        tensor = hc(tensor,
                        size=3,
                        rate=1,
                        dropout_rate=hp.dropout_rate,
                        training=training,
                        scope="HC_{}".format(i)); i += 1
    # -> (B, 7*T/r, num_lf0+num_mgc+num_bap) --> (B, 10*T/r,66)
    tensor = conv1d(tensor,
                    filters=hp.num_lf0+hp.num_mgc+hp.num_bap,
                    size=1,
                    rate=1,
                    dropout_rate=hp.dropout_rate,
                    training=training,
                    scope="C_{}".format(i)); i += 1

    for _ in range(2):
        tensor = conv1d(tensor,
                        size=1,
                        rate=1,
                        dropout_rate=hp.dropout_rate,
                        activation_fn=tf.nn.relu,
                        training=training,
                        scope="C_{}".format(i)); i += 1
    logits = conv1d(tensor,
               size=1,
               rate=1,
               dropout_rate=hp.dropout_rate,
               training=training,
               scope="C_{}".format(i))
    W = tf.nn.sigmoid(logits)
    return logits, W


class Graph:
    def __init__(self, mode="train"):
        '''
        Args:
          mode: Either "train" or "synthesize".
        '''

        # Set flag
        training = True if mode=="train" else False

        # Graph
        # Data Feeding
        ## mels: Reduced melspectrogram. (B, T/r, n_mels) float32
        ## worlds: World pad features. (B, T*10, n_fft//2+3) float32
        if mode=="train":
            self.mels, self.worlds, self.fnames, self.num_batch = get_batch()
            
        else:  # Synthesize
            self.mels = tf.placeholder(tf.float32, shape=(None, None, hp.n_mels))

        if training:
            with tf.variable_scope("Mel2World"):
                self.Z_logits, self.Z = Mel2World(self.mels, training=training)

        if not training:
            # During inference, the predicted melspectrogram values are fed.
            with tf.variable_scope("Mel2World"):
                self.Z_logits, self.Z = Mel2World(self.mels, training=training)

        with tf.variable_scope("gs"):
            self.global_step = tf.Variable(0, name='global_step', trainable=False)

        if training:
            # mag L1 loss
            self.loss_worlds = tf.reduce_mean(tf.abs(self.Z - self.worlds))

            # mag binary divergence loss
            self.loss_bd2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.Z_logits, labels=self.worlds))

            # total loss
            self.loss = self.loss_worlds + self.loss_bd2

            tf.summary.scalar('train/loss_worlds', self.loss_worlds)
            tf.summary.scalar('train/loss_bd2', self.loss_bd2)
            tf.summary.image('train/mag_gt', tf.expand_dims(tf.transpose(self.worlds[:1], [0, 2, 1]), -1))
            tf.summary.image('train/mag_hat', tf.expand_dims(tf.transpose(self.Z[:1], [0, 2, 1]), -1))

            # Training Scheme
            self.lr = learning_rate_decay(hp.lr, self.global_step)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
            tf.summary.scalar("lr", self.lr)

            ## gradient clipping
            self.gvs = self.optimizer.compute_gradients(self.loss)
            self.clipped = []
            for grad, var in self.gvs:
                grad = tf.clip_by_value(grad, -1., 1.)
                self.clipped.append((grad, var))
                self.train_op = self.optimizer.apply_gradients(self.clipped, global_step=self.global_step)

            # Summary
            self.merged = tf.summary.merge_all()


if __name__ == '__main__':

    g = Graph();print("Training Graph loaded")

    logdir = hp.logdir + "-" + str(3)
    sv = tf.train.Supervisor(logdir=logdir, save_model_secs=0, global_step=g.global_step)
    with sv.managed_session() as sess:
        while 1:
            for _ in tqdm(range(g.num_batch), total=g.num_batch, ncols=70, leave=False, unit='b'):
                gs, _ = sess.run([g.global_step, g.train_op])

                # Write checkpoint files at every 1k steps
                if gs % 1000 == 0:
                    sv.saver.save(sess, logdir + '/model_gs_{}'.format(str(gs // 1000).zfill(3) + "k")

                # break
                if gs > hp.num_iterations: break

    print("Done")
