import tensorflow as tf
FLAGS = tf.flags.FLAGS


def define_loss(action_op, actions, rewards):
    one_hot_actions = tf.one_hot(actions, FLAGS.num_actions)
    chosen_actions = tf.multiply(action_op, one_hot_actions)
    action_score_vec = tf.reduce_sum(chosen_actions, axis=1)
    loss = tf.multiply(tf.log(action_score_vec), rewards)
    loss = -tf.reduce_sum(loss)
    tf.summary.scalar("loss", loss)
    train_op = optimize(loss)
    return loss, train_op


def build_model(input):
    W_h1 = tf.get_variable('W_h1', [FLAGS.input_size, 15], initializer=tf.truncated_normal_initializer())
    b_h1 = tf.get_variable('b_h1', [15], initializer=tf.truncated_normal_initializer())
    h1 = tf.tanh(tf.matmul(input, W_h1) + b_h1)

    W_h2 = tf.get_variable('W_h2', [15, 15], initializer=tf.truncated_normal_initializer())
    b_h2 = tf.get_variable('b_h2', [15], initializer=tf.truncated_normal_initializer())
    h2 = tf.tanh(tf.matmul(h1, W_h2) + b_h2)

    W_h3 = tf.get_variable('W_h3', [15, FLAGS.num_actions], initializer=tf.truncated_normal_initializer())
    b_h3 = tf.get_variable('b_h3', [FLAGS.num_actions], initializer=tf.truncated_normal_initializer())
    h3 = tf.nn.softmax(tf.matmul(h2, W_h3) + b_h3)

    return h3


def optimize(loss):
    opt = tf.train.AdamOptimizer(learning_rate=1e-4)
    train_op = opt.minimize(loss)
    return train_op



