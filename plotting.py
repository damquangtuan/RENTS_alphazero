import numpy as np
import matplotlib.pyplot as plt
import os.path


Q = 2
K = 8000
tau = 0.08
game = 'Copy'

def get_data(filename):

	# Create some test data
	# dx = 1
	X = np.arange(0, K, 1)

	rewards = []

	for index in range(6, 50):
		flname = filename + str(index) + ".txt"
		if os.path.isfile(flname):
			text_file = open(flname, "r")
			rwds = text_file.read().split(' ')
			rwds = list(map(float, rwds))
			rewards.append(rwds)

	mean = np.quantile(rewards, 0.99, axis=0)

	# mean = np.mean(rewards, axis=0)


	std = np.std(rewards, axis=0)

	return X,mean[:K],std[:K]

tmp_tau = 0
legend = []

for index in range(8, 9):
# for tmp_tau in [3, 4, 6, 7, 8]:
# 	tmp_tau = tmp_tau + 1
	tmp_tau = 8
# 	tmp_tau = format(tmp_tau, 'f')
	filename = '../Results/' + game + '-v0_20_action_80_q_' + str(Q) + '_tau_0.0' + str(tmp_tau) + '_learning_rate_0.01_'
	# filename = '/home/damquangtuan/workspace/SparsePCL/Results/news/' + game + '-v0_20_q_2.0_tau_0.0' + str(tmp_tau) + '_learning_rate_0.01_'

	x, y, std = get_data(filename)
	plt.plot(x, y, 'o-', marker=None, markevery=1)
	plt.fill_between(x, y - std, y + std, alpha=0.3)


title = 'generaltsallis, q=' + str(Q) +  ', ' + str(game) + '-V0 - Action space=80, tau = ' + str(tau)

plt.title(title)
plt.xlabel('Training Steps')
plt.ylabel('Average Rewards')

plt.legend(('q=' + str(Q)+ ', tau=0.01', 'q=' + str(Q)+ ', tau=0.02',
        'q=' + str(Q)+ ', tau=0.03', 'q=' + str(Q)+ ', tau=0.04',
		'q=' + str(Q)+ ', tau=0.05',
		'q=' + str(Q)+ ', tau=0.06',
        'q=' + str(Q)+ ', tau=0.07',
		'q=' + str(Q)+ ', tau=0.08',
		'q=' + str(Q)+ ', tau=0.09'
        ),
       shadow=True, loc=(0.3, 0.05))

# plt.legend(('q=100, tau=0.0001', 'q=100, tau=0.0002',
#         'q=100, tau=0.0003', 'q=100, tau=0.0004', 'q=100, tau=0.0005', 'q=100, tau=0.0006',
#         'q=100, tau=0.0007', 'q=100, tau=0.0008', 'q=100, tau=0.0009'
#         ),
#        shadow=True, loc=(0.3, 0.05))

# plt.legend(('q=10, tau=0.001', 'q=10, tau=0.002',
#         'q=10, tau=0.003', 'q=10, tau=0.004', 'q=10, tau=0.005', 'q=10, tau=0.006',
#         'q=10, tau=0.007', 'q=10, tau=0.008', 'q=10, tau=0.009'
#         ),
#        shadow=True, loc=(0.3, 0.05))
plt.show()