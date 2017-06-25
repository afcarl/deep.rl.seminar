import tensorflow as tf
from model import build_model, define_loss
import gym
import numpy as np

tf.flags.DEFINE_string('env_id', "LunarLander-v2", "game name")
tf.flags.DEFINE_integer("num_actions", 4, "number of possible actions")
tf.flags.DEFINE_integer("input_size", 8, "number of possible actions")
tf.flags.DEFINE_integer("num_timesteps", 20000, "number of possible actions")
tf.flags.DEFINE_integer("num_episodes", 500000, "number of episodes")
tf.flags.DEFINE_float("lr", 1e-3, "learning rate")

FLAGS = tf.flags.FLAGS


def compute_discounted_aggregated_rewards(rewards):
    gamma = 0.99
    aggregated_rewards = []
    for i in range(len(rewards)):
        curr_step_rewards = rewards[i:]
        aggregated_reward = []
        for j in range(len(curr_step_rewards)):
            aggregated_reward.append(curr_step_rewards[j] * (pow(gamma, j)))
        aggregated_rewards.append(np.sum(aggregated_reward))
    aggregated_rewards = np.array(aggregated_rewards)
    aggregated_rewards = (aggregated_rewards - np.mean(aggregated_rewards)) / np.std(aggregated_rewards)
    return np.array(aggregated_rewards)


def main():
    observations_ph = tf.placeholder(dtype=tf.float32, shape=(None, FLAGS.input_size), name="obs")
    rewards_ph = tf.placeholder(dtype=tf.float32, shape=(None), name="rewards")
    actions_ph = tf.placeholder(dtype=tf.int32, shape=(None), name="actions")
    actions_op = build_model(observations_ph)

    loss, train_op = define_loss(actions_op, actions_ph, rewards_ph)
    env = gym.make(FLAGS.env_id)

    summary_writer = tf.summary.FileWriter('./logs/')
    summaries = tf.summary.merge_all()


    saver = tf.train.Saver()


    with tf.Session() as sess:

        sess.run(tf.initialize_all_variables())
        rewards_log = [0 for i in range(100)]
        max_avg_reward = 0
        for i_episode in range(FLAGS.num_episodes):
            observation = env.reset()
            actions = []
            obs = []
            rewards = []
            episode_len = 0
            for t in range(FLAGS.num_timesteps):
                # env.render()

                '''
                to sample from the environment, use:
                observation, reward, done, info = env.step(action)
                '''
                # YOUR CODE STARTS HERE

                # YOUR CODE ENDS HERE

                if t == FLAGS.num_timesteps - 1:
                    episode_len = t + 1
                    break

            avg_reward = log_and_compute_avg_reward(rewards, rewards_log)
            actions, obs, rewards = preprocessing(actions, obs, rewards)
            loss_val, _, summary_str = sess.run([loss, train_op, summaries], feed_dict={observations_ph: obs,
                                                                                        actions_ph: actions,
                                                                                        rewards_ph: rewards})

            summary_writer.add_summary(summary_str, global_step=i_episode)
            if avg_reward > max_avg_reward and avg_reward > 100:
                print "Saved model with avg reward: %s" % avg_reward
                max_avg_reward = avg_reward
                saver.save(sess, save_path='./logs/ckp_%s' % max_avg_reward, global_step=i_episode)

            if i_episode % 1000 == 0:
                print "Num episode: %s, length: %s, loss: %s, reward: %s" % (
                i_episode, episode_len, loss_val, rewards_log[-1])

    main()


def log_and_compute_avg_reward(rewards, rewards_log):
    rewards_log.append(np.sum(rewards))
    rewards_log.pop(0)
    avg_reward = np.mean(rewards_log)
    return avg_reward


def preprocessing(actions, obs, rewards):
    rewards = compute_discounted_aggregated_rewards(rewards)
    return actions, obs, rewards

main()