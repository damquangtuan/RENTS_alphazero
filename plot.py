import numpy as np
import matplotlib.pyplot as plt
import os

params = {'legend.fontsize': 14,
         'figure.figsize': (4.25, 4.0),
          # 'figure.figsize': (9.75, 2.8125),
         'axes.labelsize': 14,
         'axes.titlesize':13,
         'xtick.labelsize':14,
         'ytick.labelsize':14}

def read_data(filename):

    file = open(filename, "r")

    sims = []
    rewards = []
    variances = []

    cout = -1

    for line in file:
        fields = line.split()
        cout = cout + 1
        if cout > 0:
            sim = fields[0]
            sims.append(float(sim))
            reward = fields[0]
            rewards.append(float(reward))

    rewards = rewards[0:499]

    # return sims, np.power(np.e, rewards), np.power(np.e, variances)

    # return sims, np.power(np.e, rewards), variances

    return sims, rewards, variances

def get_data(filename):

    # Create some test data
    # dx = 1
    K = 476
    X = np.arange(0, K, 1)

    rewards = []

    for index in range(1, 6):
        flname = filename + str(index) + ".txt"
        reward = []
        if os.path.isfile(flname):
            text_file = open(flname, "r")
            for line in text_file:
                rwds = line
                rwds = float(rwds)
                reward.append(rwds)

        rewards.append(reward)

    # mean = np.quantile(rewards, 0.95, axis=0)

    mean = np.mean(rewards, axis=0)
    std = np.std(rewards, axis=0)


    return X, mean[:K], std[:K]

def Acrobot():
    plt.title("Acrobot-v1, Number of run = 5")
    ments_sims, ments_rewards, ments_std = get_data("./data/Acrobot-v1_ments_")
    plt.xlabel("Episole")
    plt.ylabel("Total Rewards")
    plt.tick_params(axis="y", labelcolor="b")

    plt.plot(ments_sims, ments_rewards, 'o-', marker=None, markevery=1)
    plt.fill_between(ments_sims, np.array(ments_rewards) - np.array(ments_std), np.array(ments_rewards) + np.array(ments_std), alpha=0.3)
    #######################################################################
    # sims, rewards, std = read_data("output.txt_rocksample_7_8_numrun_100_h_3_sim_20_rolloutknowledge_1")
    # plt.xlabel("Simulation")
    # plt.ylabel("Discounted Reward")
    # plt.tick_params(axis="y", labelcolor="b")
    #
    # plt.plot(sims, rewards, 'o-', marker=None, markevery=1)
    # plt.fill_between(sims, np.array(rewards) - np.array(std), np.array(rewards) + np.array(std), alpha=0.3)
    #######################################################################
    # sims, rewards, std = read_data("output.txt_rocksample_7_8_numrun_100_h_4_sim_18_rolloutknowledge_1")
    # plt.xlabel("Simulation")
    # plt.ylabel("Discounted Reward")
    # plt.tick_params(axis="y", labelcolor="b")
    #
    # plt.plot(sims, rewards, 'o-', marker=None, markevery=1)
    # plt.fill_between(sims, np.array(rewards) - np.array(std), np.array(rewards) + np.array(std), alpha=0.3)
    #######################################################################

    rents_sims, rents_rewards, rents_std = get_data("./data/Acrobot-v1_rents_")
    plt.xlabel("Episole")
    plt.ylabel("Total Rewards")
    plt.tick_params(axis="y", labelcolor="b")

    plt.plot(rents_sims, rents_rewards, 'o-', marker=None, markevery=1)
    plt.fill_between(rents_sims, np.array(rents_rewards) - np.array(rents_std), np.array(rents_rewards) + np.array(rents_std), alpha=0.3)

    plt.legend(('MENTS', 'RENTS'), shadow=True, loc=(0.3, 0.05))

    plt.show()

def CartPole():
    plt.title("CartPole-v1, Number of run = 5")
    ments_sims, ments_rewards, ments_std = get_data("./data/CartPole-v1_ments_")
    plt.xlabel("Episole")
    plt.ylabel("Total Rewards")
    plt.tick_params(axis="y", labelcolor="b")

    plt.plot(ments_sims, ments_rewards, 'o-', marker=None, markevery=1)
    plt.fill_between(ments_sims, np.array(ments_rewards) - np.array(ments_std), np.array(ments_rewards) + np.array(ments_std), alpha=0.3)
    #######################################################################
    # sims, rewards, std = read_data("output.txt_rocksample_7_8_numrun_100_h_3_sim_20_rolloutknowledge_1")
    # plt.xlabel("Simulation")
    # plt.ylabel("Discounted Reward")
    # plt.tick_params(axis="y", labelcolor="b")
    #
    # plt.plot(sims, rewards, 'o-', marker=None, markevery=1)
    # plt.fill_between(sims, np.array(rewards) - np.array(std), np.array(rewards) + np.array(std), alpha=0.3)
    #######################################################################
    # sims, rewards, std = read_data("output.txt_rocksample_7_8_numrun_100_h_4_sim_18_rolloutknowledge_1")
    # plt.xlabel("Simulation")
    # plt.ylabel("Discounted Reward")
    # plt.tick_params(axis="y", labelcolor="b")
    #
    # plt.plot(sims, rewards, 'o-', marker=None, markevery=1)
    # plt.fill_between(sims, np.array(rewards) - np.array(std), np.array(rewards) + np.array(std), alpha=0.3)
    #######################################################################

    rents_sims, rents_rewards, rents_std = get_data("./data/CartPole-v1_rents_")
    plt.xlabel("Episole")
    plt.ylabel("Total Rewards")
    plt.tick_params(axis="y", labelcolor="b")

    plt.plot(rents_sims, rents_rewards, 'o-', marker=None, markevery=1)
    plt.fill_between(rents_sims, np.array(rents_rewards) - np.array(rents_std), np.array(rents_rewards) + np.array(rents_std), alpha=0.3)

    plt.legend(('MENTS', 'RENTS'), shadow=True, loc=(0.3, 0.05))

    plt.show()

if __name__ == '__main__':
    Acrobot()
    # CartPole()
    # get_data("./data/Acrobot-v1_ments_")