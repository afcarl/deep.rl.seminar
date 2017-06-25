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
    '''
    Your NN should have 2 FC layer with TANH activation.
    Each layer should contain ~15 neurons.
    :param input: tensor of size 8
    :returns distribution over actions. in this case tensor in size 4
    '''
    # YOUR CODE STARTS HERE

    # YOUR CODE ENDS HERE
    return None


def optimize(loss):
    opt = tf.train.AdamOptimizer(learning_rate=1e-4)
    train_op = opt.minimize(loss)
    return train_op



