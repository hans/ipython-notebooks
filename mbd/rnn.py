import sys

from tf import util as tf_util

import gflags
import numpy as np
import tensorflow as tf
import tables


FLAGS = gflags.FLAGS

gflags.DEFINE_string("data_file", None, "")

gflags.DEFINE_integer("n_channels", 1, "")
gflags.DEFINE_integer("n_classes", 11, "")
gflags.DEFINE_integer("seq_length", 1024, "")

gflags.DEFINE_integer("hidden_dim", 128, "")

gflags.DEFINE_integer("batch_size", 256, "")
gflags.DEFINE_float("lr", 0.1, "")
gflags.DEFINE_float("momentum", 0.9, "")


def model(X, lengths, y):
  batch_size = tf.shape(X[0])[0]
  y_mat = tf_util.convert_labels_to_onehot(y + 1, FLAGS.batch_size,
                                           FLAGS.n_classes)
  # TODO normalize

  cell = tf.nn.rnn_cell.GRUCell(FLAGS.hidden_dim)
  init_state = cell.zero_state(batch_size, tf.float32)
  outputs, states = tf.nn.rnn(cell, X, init_state)

  final_out = outputs[-1]

  W = tf.get_variable("W", (cell.output_size, FLAGS.n_classes))
  logits = tf.matmul(final_out, W)

  loss = tf.nn.softmax_cross_entropy_with_logits(logits, y_mat)
  loss = tf.reduce_mean(loss)

  return logits, loss


def data_iter():
  with tables.open_file(FLAGS.data_file) as h5file:
    X, y, lengths = h5file.root.X, h5file.root.y, h5file.root.lengths

    test_idxs = np.random.choice(len(X), replace=False, size=int(len(X) * 0.2))
    train_idxs = np.array(list(set(range(len(X))) - set(test_idxs)))

    while True:
      batch_idxs = np.random.choice(train_idxs, replace=False,
                                    size=FLAGS.batch_size)

      # X_batch = np.transpose(X[cursor:cursor + FLAGS.batch_size], (2, 0, 1))
      # y_batch = y[cursor:cursor + FLAGS.batch_size]
      # lengths_batch = lengths[cursor:cursor + FLAGS.batch_size]
      X_batch = np.transpose(X[batch_idxs, :, :], (2, 0, 1))
      y_batch = y[batch_idxs]
      lengths_batch = lengths[batch_idxs]

      # test_offset = test_start + test_cursor
      # X_test = np.transpose(X[test_offset:test_offset + FLAGS.batch_size], (2, 0, 1))
      # y_test = y[test_offset:test_offset + FLAGS.batch_size]
      # lengths_test = lengths[test_offset:test_offset + FLAGS.batch_size]

      test_batch_idxs = np.random.choice(test_idxs, replace=False,
                                         size=FLAGS.batch_size)
      X_test = np.transpose(X[test_batch_idxs, :, :], (2, 0, 1))
      y_test = y[test_batch_idxs]
      lengths_test = lengths[test_batch_idxs]

      yield (X_batch, y_batch, lengths_batch), (X_test, y_test, lengths_test)


def main():
  X = [tf.placeholder(tf.float32, shape=(None, FLAGS.n_channels), name="X%i" % t)
       for t in range(FLAGS.seq_length)]
  lengths = tf.placeholder(tf.int32, shape=(None,), name="lengths")
  y = tf.placeholder(tf.int32, shape=(None,), name="y")

  logits, loss = model(X, lengths, y)

  minimizer = tf.train.MomentumOptimizer(FLAGS.lr, FLAGS.momentum)
  train_op = minimizer.minimize(loss)

  sess = tf.Session()
  sess.run(tf.initialize_all_variables())
  for train_batch, test_batch in data_iter():
    X_batch, y_batch, lengths_batch = train_batch
    feed_dict = {X[t]: X_batch[t] for t in range(FLAGS.seq_length)}
    feed_dict[y] = y_batch

    _, loss_t = sess.run([train_op, loss], feed_dict)

    #############

    X_test, y_test, lengths_test = test_batch
    feed_dict = {X[t]: X_test[t] for t in range(FLAGS.seq_length)}
    feed_dict[y] = y_test
    test_loss_t = sess.run(loss, feed_dict)

    print "%10f\t%10f" % (loss_t, test_loss_t)


if __name__ == "__main__":
  FLAGS(sys.argv)
  main()
