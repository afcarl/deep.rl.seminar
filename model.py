import tensorflow as tf
FLAGS = tf.flags.FLAGS


def define_loss(action_op, actions, rewards):
    '''
    Compute loss given sampled actions and rewards
    :param action_op - TF op computing distribution over actions
    :param actions - list of sampled actions
    :param rewards - list of sampled rewards
    :returns
    :param loss_op - operator computing loss
    :param train_op - training optimizer op
    '''
    # YOUR CODE STARTS HERE
    loss = 0
    # YOUR CODE ENDS HERE

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
