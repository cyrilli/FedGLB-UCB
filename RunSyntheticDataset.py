import numpy as np
import random
import datetime
import os.path
import matplotlib.pyplot as plt
import argparse

import time
from conf import save_address

from lib.One_UCB_GLM import One_UCB_GLM
from lib.N_UCB_GLM import N_UCB_GLM
from lib.N_ONS_GLM import N_ONS_GLM
from lib.DisLinUCB import DisLinUCB
from lib.FedGLB_UCB import FedGLB_UCB
from lib.FedGLB_UCB_v1 import FedGLB_UCB_v1
from lib.FedGLB_UCB_v2 import FedGLB_UCB_v2
from lib.FedGLB_UCB_v3 import FedGLB_UCB_v3

def sigmoid(x):
    return 1/(1+np.exp(-x))

def dsigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))

class simulateOnlineData(object):
	def __init__(self, context_dimension, testing_iterations, n_users, theta_star,
				 article_feature_matrix, poolArticleSize, namelabel, plot=True):
		self.namelabel = namelabel
		self.context_dimension = context_dimension
		self.testing_iterations = testing_iterations
		self.n_users = n_users
		self.theta_star = theta_star
		self.article_feature_matrix = article_feature_matrix
		self.n_articles = len(article_feature_matrix)
		self.batchSize = 1

		if poolArticleSize is None:
			self.poolArticleSize = len(self.articles)
		else:
			self.poolArticleSize = poolArticleSize

		self.plot = plot

	def getTheta(self):
		Theta = np.zeros(shape = (self.context_dimension, len(self.users)))
		for i in range(len(self.users)):
			Theta.T[i] = self.users[i].theta
		return Theta
	
	def getReward(self, theta_star, pickedArticleFeatureVec):
		mean_reward = self.logistic(np.dot(theta_star, pickedArticleFeatureVec))
		return mean_reward

	def GetOptimalReward(self, theta_star, articleFeatureMatrix):		
		meanRewards = np.dot(articleFeatureMatrix, theta_star)
		assert meanRewards.shape == (self.poolArticleSize, )
		return self.logistic(np.max(meanRewards))

	def logistic(self, x):
		return 1/(1+np.exp(-x))

	def getL2Diff(self, x, y):
		return np.linalg.norm(x-y) # L2 norm

	def regulateArticlePool(self):
		# Randomly sample articles
		idx = np.random.randint(self.n_articles, size=self.poolArticleSize)
		return self.article_feature_matrix[idx, :]

	def runAlgorithms(self, algorithms):
		self.startTime = datetime.datetime.now()
		timeRun = "%0.2d_%0.2d_%0.2d_%0.2d" % (self.startTime.day, self.startTime.hour, self.startTime.minute, self.startTime.second)
		filenameWriteRegret = os.path.join(save_address, 'AccRegret' + '_' + timeRun + self.namelabel + '.csv')
		filenameWriteCommCost = os.path.join(save_address, 'AccCommCost' + '_' + timeRun + self.namelabel+ '.csv')
		filenameWritePara = os.path.join(save_address, 'ParameterEstimation' + '_' + timeRun + self.namelabel + '.csv')

		tim_ = []
		BatchCumlateRegret = {}
		CommCostList = {}
		AlgRegret = {}
		ThetaDiffList = {}
		ThetaDiff = {}
		
		# Initialization
		for alg_name, alg in algorithms.items():
			AlgRegret[alg_name] = []
			CommCostList[alg_name] = []
			BatchCumlateRegret[alg_name] = []
			if alg.CanEstimateUserPreference:
				ThetaDiffList[alg_name] = []

		with open(filenameWriteRegret, 'w') as f:
			f.write('Time(Iteration)')
			f.write(',' + ','.join([str(alg_name) for alg_name in algorithms.keys()]))
			f.write('\n')

		with open(filenameWriteCommCost, 'w') as f:
			f.write('Time(Iteration)')
			f.write(',' + ','.join([str(alg_name) for alg_name in algorithms.keys()]))
			f.write('\n')

		with open(filenameWritePara, 'w') as f:
			f.write('Time(Iteration)')
			f.write(','+ ','.join([str(alg_name)+'Theta' for alg_name in ThetaDiffList.keys()]))
			f.write('\n')

		for iter_ in range(self.testing_iterations):
			# prepare to record theta estimation error
			for alg_name, alg in algorithms.items():
				if alg.CanEstimateUserPreference:
					ThetaDiff[alg_name] = 0
			for uid in range(self.n_users):
				candidateArticleFeatureMatrix = self.regulateArticlePool()
				#get optimal mean reward
				OptimalMeanReward = self.GetOptimalReward(self.theta_star, candidateArticleFeatureMatrix)

				for alg_name, alg in algorithms.items():
					pickedArticleFeatureVec, _ = alg.decide(candidateArticleFeatureMatrix, uid)
					meanReward = self.getReward(self.theta_star, pickedArticleFeatureVec)
					reward = np.random.binomial(1, p=meanReward)
					alg.updateParameters(pickedArticleFeatureVec, reward, uid)

					regret = OptimalMeanReward - meanReward  # pseudo regret, difference between mean reward
					AlgRegret[alg_name].append(regret)

					# Update parameter estimation record
					if alg.CanEstimateUserPreference:
						ThetaDiff[alg_name] += self.getL2Diff(self.theta_star, alg.getTheta(uid))

			for alg_name, alg in algorithms.items():
				if alg.CanEstimateUserPreference:
					ThetaDiffList[alg_name] += [ThetaDiff[alg_name] / self.n_users]

			if iter_%self.batchSize == 0:
				print("Iteration %d"%iter_, " Elapsed time", datetime.datetime.now() - self.startTime)
				
				tim_.append(iter_)
				for alg_name, alg in algorithms.items():
					cumRegret = sum(AlgRegret[alg_name])
					BatchCumlateRegret[alg_name].append(cumRegret)
					CommCostList[alg_name].append(alg.totalCommCost)
					print("{0: <16}: cum_regret {1}, cum_comm {2}".format(alg_name, cumRegret, alg.totalCommCost))
				with open(filenameWriteRegret, 'a+') as f:
					f.write(str(iter_))
					f.write(',' + ','.join([str(BatchCumlateRegret[alg_name][-1]) for alg_name in algorithms.keys()]))
					f.write('\n')
				with open(filenameWriteCommCost, 'a+') as f:
					f.write(str(iter_))
					f.write(',' + ','.join([str(CommCostList[alg_name][-1]) for alg_name in algorithms.keys()]))
					f.write('\n')
				with open(filenameWritePara, 'a+') as f:
					f.write(str(iter_))
					f.write(','+ ','.join([str(ThetaDiffList[alg_name][-1]) for alg_name in ThetaDiffList.keys()]))
					f.write('\n')

		if (self.plot==True):
			# # plot the results
			fig, axa = plt.subplots(2, 1, sharex='all')
			fig.subplots_adjust(hspace=0) # Remove horizontal space between axes

			print("=====Regret=====")
			for alg_name in algorithms.keys():
				axa[0].plot(tim_, BatchCumlateRegret[alg_name],label = alg_name)
				print('%s: %.2f' % (alg_name, BatchCumlateRegret[alg_name][-1]))
			axa[0].legend(loc='upper left',prop={'size':9})
			axa[0].set_xlabel("Iteration")
			axa[0].set_ylabel("Accumulative Regret")

			print("=====Comm Cost=====")
			for alg_name in algorithms.keys():
				axa[1].plot(tim_, CommCostList[alg_name],label = alg_name)
				print('%s: %.2f' % (alg_name, CommCostList[alg_name][-1]))
			axa[1].set_xlabel("Iteration")
			axa[1].set_ylabel("Communication Cost")

			plt.savefig(os.path.join(save_address, "PlotRegretComm" + "_" + timeRun + self.namelabel + '.png'), dpi=300, bbox_inches='tight', pad_inches=0.0)
			# plt.show()

		finalRegret = {}
		for alg_name in algorithms.keys():
			finalRegret[alg_name] = BatchCumlateRegret[alg_name][:-1]
		return finalRegret

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description = '')
	parser.add_argument('--N', type=int, default=200, dest='N', help='Set number of users.')
	parser.add_argument('--T', type=int, default=2000, dest='T', help='Set length of time horizon.')
	parser.add_argument('--d', type=int, default=10, dest='d', help='Set dimension of context features.')
	parser.add_argument('--namelabel', default=None ,dest='namelabel', help='Name')
	args = parser.parse_args()

	## Environment Settings ##
	args = parser.parse_args()
	context_dimension = int(args.d)
	testing_iterations = int(args.T)
	n_users = int(args.N)
	if args.namelabel is not None:
		namelabel = str(args.namelabel)
	else:
		namelabel = 'T{}N{}d{}'.format(testing_iterations, n_users, context_dimension)

	n_articles = 1000
	poolArticleSize = 25


	## Simulate users and articles ##
	# generate theta_star
	theta_star = np.random.normal(size=context_dimension)
	l2_norm = np.linalg.norm(theta_star, ord=2)
	theta_star = theta_star/l2_norm

	# generate context features
	# standard d-dimensional basis (with a bias term)
	basis = np.eye(context_dimension)
	basis[:, -1] = 1
	# arm features in a unit (d - 2)-sphere
	feature_matrix = np.random.randn(n_articles, context_dimension - 1)
	feature_matrix /= np.sqrt(np.square(feature_matrix).sum(axis=1))[:, np.newaxis]
	feature_matrix = np.hstack((feature_matrix, np.ones((n_articles, 1))))  # bias term
	feature_matrix[: basis.shape[0], :] = basis

	simExperiment = simulateOnlineData( context_dimension=context_dimension,
										testing_iterations=testing_iterations,
										n_users=n_users,
										theta_star = theta_star,
										article_feature_matrix=feature_matrix,
										poolArticleSize=poolArticleSize,
										namelabel=namelabel
										)

	## Setup Bandit Algorithms ##
	algorithms = {}

	lambda_ = 2
	delta = 1e-2
	alpha = 1
	S = 1
	R = 0.5
	c_mu = dsigmoid(S * 1)

	global_update_period = round(np.sqrt(n_users*testing_iterations))

	algorithms['FedGLB_UCB'] = FedGLB_UCB(dimension=context_dimension, lambda_=lambda_, threshold=testing_iterations/(n_users*context_dimension*np.log(n_users*testing_iterations)), n_users=n_users, c_mu=c_mu, delta=delta, S=S, R=R, alpha=None, alpha_t_scaling=0.1, init_x=np.zeros(context_dimension), max_iters=None)
	algorithms['FedGLB_UCB_v1'] = FedGLB_UCB_v1(dimension=context_dimension, lambda_=lambda_, sync_period=global_update_period, delta=delta, alpha = None, alpha_t_scaling=1.0, R=R, S=S, c_mu=c_mu, max_iters=None, init_x=np.zeros(context_dimension))
	algorithms['FedGLB_UCB_v2'] = FedGLB_UCB_v2(dimension=context_dimension, lambda_=lambda_, sync_period=global_update_period, n_users=n_users, c_mu=c_mu, delta=delta, S=S, R=R, alpha=None, alpha_t_scaling=0.1, init_x=np.zeros(context_dimension), max_iters=None)
	algorithms['FedGLB_UCB_v3'] = FedGLB_UCB_v3(dimension=context_dimension, lambda_=lambda_, alpha=None, sync_period=global_update_period, n_users=n_users, c_mu=c_mu, S=S, R=R, delta=delta, lr=None, alpha_t_scaling=1e-1)
	algorithms['One_UCB_GLM'] = One_UCB_GLM(dim=context_dimension, horizon=testing_iterations, numUsers=n_users, lambda_=lambda_, delta=delta, alpha=None, alpha_t_scaling=1.0, max_iters=5, R=R)
	algorithms['N_UCB_GLM'] = N_UCB_GLM(dim=context_dimension, horizon=testing_iterations, numUsers=n_users, lambda_=lambda_, c_mu=c_mu, delta=delta, alpha=None, alpha_t_scaling=1.0, max_iters=100, R=R)
	algorithms['N_ONS_GLM'] = N_ONS_GLM(dimension=context_dimension, lambda_=lambda_, alpha=None, alpha_t_scaling=0.1, c_mu=c_mu, delta=delta, S=S, R=R)
	
	## Run Simulation ##
	print("Starting for ", simExperiment.namelabel)
	simExperiment.runAlgorithms(algorithms)