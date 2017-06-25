import tensorflow as tf
from model import build_model, define_loss
import gym
import numpy as np

tf.flags.DEFINE_string('env_id', "LunarLander-v2", "game name")
tf.flags.DEFINE_integer("num_actions", 4, "number of possible actions")
tf.flags.DEFINE_integer("input_size", 8, "number of possible actions")
tf.flags.DEFINE_integer("num_timesteps", 5000, "number of possible actions")
tf.flags.DEFINE_integer("num_episodes", 30000, "number of episodes")

FLAGS = tf.flags.FLAGS



observations_ph = tf.placeholder(dtype=tf.float32, shape=(None, FLAGS.input_size), name="obs")
rewards_ph = tf.placeholder(dtype=tf.float32, shape=(None), name="rewards")
actions_ph = tf.placeholder(dtype=tf.int32, shape=(None), name="actions")
actions_op = build_model(observations_ph)

loss, train_op = define_loss(actions_op, actions_ph, rewards_ph)
env = gym.make(FLAGS.env_id)

summary_writer = tf.summary.FileWriter('./logs/')
summaries = tf.summary.merge_all()


def compute_discounted_aggregated_rewards(rewards):
    gamma = 0.99
    aggregated_rewards = []
    for i in range(len(rewards)):
        curr_step_rewards = rewards[i:]
        aggregated_reward = []
        for j in range(len(curr_step_rewards)):
            aggregated_reward.append(curr_step_rewards[j]*(pow(gamma, j)))
        aggregated_rewards.append(np.sum(aggregated_reward))
    aggregated_rewards = np.array(aggregated_rewards)
    aggregated_rewards = (aggregated_rewards - np.mean(aggregated_rewards))/np.std(aggregated_rewards)
    return aggregated_rewards

saver = tf.train.Saver()
with tf.Session() as sess:

    sess.run(tf.initialize_all_variables())
    saver.restore(sess, '/Users/amirbar/Repositories/adv.ml.hw4/logs/ckp_218.433660562-13469')
    for i_episode in range(FLAGS.num_episodes):
        observation = env.reset()
        actions = []
        obs = []
        rewards = []
        episode_len = 0
        for t in range(FLAGS.num_timesteps):
            env.render()
            curr = []
            obs.append(observation)

            predicted_actions = sess.run(actions_op, feed_dict={observations_ph: np.expand_dims(observation, 0)})
            action = np.random.choice(np.arange(0, FLAGS.num_actions), p=np.squeeze(predicted_actions))
            actions.append(action)

            observation, reward, done, info = env.step(action)
            rewards.append(reward)

            if done or t == FLAGS.num_timesteps-1:
                episode_len = t+1
                # print("Episode finished after {} timesteps".format(t + 1))
                break
