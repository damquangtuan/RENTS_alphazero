import numpy as np
import os
import GPy as gpy
from GPyOpt.methods import BayesianOptimization
import subprocess
import shutil


def read_data(filename):

    file = open(filename, "r")
    rewards = []
    cout = -1
    for line in file:
        fields = line.split()
        cout = cout + 1
        if cout > 0:
            reward = fields[0]
            rewards.append(float(reward))

    rewards = rewards[0:1]
    return np.mean(rewards)


def run_experiment(args):
    print(args)
    # Disassemble the iinput
    res_total = []
    for i in range(args.shape[0]):
        v_tau = args[i, 0]
        p_tau = args[i, 1]
        epsilon = args[i, 2]

        # Create the log directory
        subprocess.check_call(
            "/home/tuandam/.conda/envs/python37/bin/python3.7 /home/tuandam/workspace/alphazero_singleplayer/alphazero_ments.py --game=BreakoutNoFrameskip-v4 "
            + "--v_tau=" + str(v_tau) + " --p_tau=" + str(p_tau) + " --epsilon=" + str(epsilon), shell=True)

        # Collect the result
        filename = os.getcwd() + '/logs/BreakoutNoFrameskip-v4' + '_ments.txt_' + str(v_tau) + '_' + str(p_tau) + '_' + str(epsilon)

        print("filename: " + str(filename))

        res = read_data(filename)
        res_total.append(res)

    return np.array(res_total)


def load_initial_experiments():
    xs = []
    ys = []

    for log_dir in os.listdir("./logs"):
        prefix = 'BreakoutNoFrameskip-v4' + '_ments.txt_'
        if log_dir.startswith(prefix):
            v_tau, p_tau, epsilon = log_dir[len(prefix):].split("_")
            v_tau = float(v_tau)
            p_tau = float(p_tau)
            epsilon = float(epsilon)
            xs.append([v_tau, p_tau, epsilon])
            ys.append(run_experiment(np.array([xs[-1]])))

    return np.array(xs), np.array(ys)


if __name__ == "__main__":
    np.random.seed(0)
    bds = [{'name': 'v_tau', 'type': 'continuous', 'domain': (.01, 2.)},
           {'name': 'p_tau', 'type': 'continuous', 'domain': (.01, 2.)},
           {'name': 'epsilon', 'type': 'continuous', 'domain': (.1, 5.)}]

    xs, ys = load_initial_experiments()

    optimizer = BayesianOptimization(f=run_experiment, domain=bds, model_type='GP_MCMC',
                                     acquisition_type ='MPI_MCMC', exact_feval=True, maximize=True, batch_size=1,
                                     X=xs, Y=ys)

    optimizer.run_optimization(max_iter=1000)