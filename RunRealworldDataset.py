import copy
import numpy as np
import random
from random import sample, shuffle, choice
from scipy.sparse import csgraph
import datetime
import os.path
import json
import matplotlib.pyplot as plt
import argparse

# local address to different datasets
from conf import *

# real dataset
from dataset_utils.LastFM_util_functions_2 import readFeatureVectorFile, parseLine

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

class Article():
    def __init__(self, aid, FV=None):
        self.article_id = aid
        self.contextFeatureVector = FV
        self.featureVector = FV


class experimentOneRealData(object):
    def __init__(self, namelabel, dataset, context_dimension, batchSize=1, Write_to_File=False):

        self.namelabel = namelabel
        self.dataset = dataset
        self.context_dimension = context_dimension
        self.Write_to_File = Write_to_File
        self.batchSize = batchSize

        self.relationFileName = MovieLens_relationFileName
        self.address = MovieLens_address
        self.save_address = MovieLens_save_address
        FeatureVectorsFileName = MovieLens_FeatureVectorsFileName
        self.event_fileName = self.address + "/randUserOrderedTime_N37_ObsMoreThan2500_PosOverThree.dat"
        # Read Feature Vectors from File
        self.FeatureVectors = readFeatureVectorFile(FeatureVectorsFileName)
        self.articlePool = []

    def batchRecord(self, iter_):
        print("Iteration %d" % iter_, "Pool", len(self.articlePool), " Elapsed time",
              datetime.datetime.now() - self.startTime)

    def runAlgorithms(self, algorithms, startTime):
        self.startTime = startTime
        timeRun = self.startTime.strftime('_%m_%d_%H_%M_%S')

        filenameWriteReward = os.path.join(self.save_address, 'AccReward' + str(self.namelabel) + timeRun + '.csv')

        end_num = 0
        while os.path.exists(filenameWriteReward):
            filenameWriteReward = os.path.join(self.save_address,'AccReward' + str(self.namelabel) + timeRun + str(end_num) + '.csv')
            end_num += 1

        filenameWriteCommCost = os.path.join(self.save_address, 'AccCommCost' + str(self.namelabel) + timeRun + '.csv')
        end_num = 0
        while os.path.exists(filenameWriteCommCost):
            filenameWriteCommCost = os.path.join(self.save_address,'AccCommCost' + str(self.namelabel) + timeRun + str(end_num) + '.csv')
            end_num += 1
        tim_ = []
        AlgReward = {}
        BatchCumlateReward = {}
        CommCostList = {}
        AlgReward["random"] = []
        BatchCumlateReward["random"] = []
        for alg_name, alg in algorithms.items():
            AlgReward[alg_name] = []
            BatchCumlateReward[alg_name] = []
            CommCostList[alg_name] = []

        if self.Write_to_File:
            with open(filenameWriteReward, 'w') as f:
                f.write('Time(Iteration)')
                f.write(',' + ','.join([str(alg_name) for alg_name in algorithms.keys()]))
                f.write('\n')

            with open(filenameWriteCommCost, 'w') as f:
                f.write('Time(Iteration)')
                f.write(',' + ','.join([str(alg_name) for alg_name in algorithms.keys()]))
                f.write('\n')

        userIDSet = set()
        with open(self.event_fileName, 'r') as f:
            f.readline()
            iter_ = 0
            for _, line in enumerate(f, 1):
                userID, _, pool_articles = parseLine(line)
                if userID not in userIDSet:
                    userIDSet.add(userID)
                # ground truth chosen article
                article_id_chosen = int(pool_articles[0])
                # Construct arm pool
                self.article_pool = []
                for article in pool_articles:
                    article_id = int(article.strip(']'))
                    article_featureVector = self.FeatureVectors[article_id]
                    article_featureVector = np.array(article_featureVector, dtype=float)
                    article_featureVector = np.concatenate((article_featureVector, [1.0]))
                    assert type(article_featureVector) == np.ndarray
                    assert article_featureVector.shape == (self.context_dimension,)
                    self.article_pool.append(Article(article_id, article_featureVector))

                candidateArticleFeatureMatrix = np.ndarray([0, self.context_dimension])
                for article in self.article_pool:
                    candidateArticleFeatureMatrix = np.concatenate((candidateArticleFeatureMatrix, [article.featureVector]), axis=0)
                assert candidateArticleFeatureMatrix.shape == (len(pool_articles), self.context_dimension)

                # Random strategy
                RandomPicked = choice(self.article_pool)
                if RandomPicked.article_id == article_id_chosen:
                    reward = 1
                else:
                    reward = 0 
                AlgReward["random"].append(reward)

                for alg_name, alg in algorithms.items():
                    # Observe the candiate arm pool and algoirhtm makes a decision
                    pickedArticleVec, pickedArticle = alg.decide(candidateArticleFeatureMatrix, userID)
                    # Get the feedback by looking at whether the selected arm by alg is the same as that of ground truth
                    if self.article_pool[pickedArticle].article_id == article_id_chosen:
                        reward = 1
                    else:
                        reward = 0

                    alg.updateParameters(pickedArticleVec, reward, userID)
                    # Record the reward
                    AlgReward[alg_name].append(reward)
                    CommCostList[alg_name].append(alg.totalCommCost)

                if iter_ % self.batchSize == 0:
                    self.batchRecord(iter_)
                    tim_.append(iter_)
                    BatchCumlateReward["random"].append(sum(AlgReward["random"]))
                    print("{0: <16}: cum_reward {1}, cum_comm {2}".format("random", BatchCumlateReward["random"][-1], 0))
                    for alg_name in algorithms.keys():
                        BatchCumlateReward[alg_name].append(sum(AlgReward[alg_name]))

                        print("{0: <16}: cum_reward {1}, cum_comm {2}".format(alg_name, BatchCumlateReward[alg_name][-1], CommCostList[alg_name][-1]))

                    if self.Write_to_File:
                        with open(filenameWriteReward, 'a+') as f:
                            f.write(str(iter_))
                            f.write(',' + ','.join([str(BatchCumlateReward[alg_name][-1]) for alg_name in
                                                    list(algorithms.keys()) + ["random"]]))
                            f.write('\n')
                        with open(filenameWriteCommCost, 'a+') as f:
                            f.write(str(iter_))
                            f.write(',' + ','.join([str(CommCostList[alg_name][-1]) for alg_name in algorithms.keys()]))
                            f.write('\n')
                iter_ += 1

        for alg_name in algorithms.keys():
            print('%s: %.2f' % (alg_name, BatchCumlateReward[alg_name][-1]))

        return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--namelabel', dest='namelabel', help='Name')

    args = parser.parse_args()
    namelabel = str(args.namelabel)
    dataset = "MovieLens"
    # Configuration about the environment
    Write_to_File = True

    # environment parameters
    n_users = 37
    testing_iterations = 121934
    # item number 26567
    context_dimension = 26  # feature dimension + one bias term

    # Algorithm parameters
    lambda_ = 0.2  # regularization in ridge regression
    delta = 1e-2
    alpha = 1
    S = 1
    R = 0.5
    c_mu = dsigmoid(S * 1)
    alpha = 0.3
    alpha_linear = 0.3

    realExperiment = experimentOneRealData(namelabel=namelabel,
                                           dataset=dataset,
                                           context_dimension=context_dimension,
                                           Write_to_File=Write_to_File)

    # set up algorithms
    print("Starting for {}, context dimension {}".format(realExperiment.dataset, realExperiment.context_dimension))
    algorithms = {}

    global_update_period = np.sqrt(testing_iterations)

    algorithms['FedGLB_UCB'] = FedGLB_UCB(dimension=context_dimension, lambda_=lambda_, threshold=2, n_users=n_users, c_mu=c_mu, delta=delta, S=S, R=R, alpha=alpha, alpha_t_scaling=1.0, init_x=None, max_iters=1000)
    algorithms['FedGLB_UCB_v1'] = FedGLB_UCB_v1(dimension=context_dimension, lambda_=lambda_, sync_period=global_update_period, delta=delta, alpha = alpha, alpha_t_scaling=1.0, R=R, S=S, c_mu=c_mu, max_iters=1000, init_x=None)
    algorithms['FedGLB_UCB_v2'] = FedGLB_UCB_v2(dimension=context_dimension, lambda_=lambda_, sync_period=global_update_period, n_users=n_users, c_mu=c_mu, delta=delta, S=S, R=R, alpha=alpha, alpha_t_scaling=1.0, init_x=None, max_iters=1000)
    algorithms['FedGLB_UCB_v3'] = FedGLB_UCB_v3(dimension=context_dimension, lambda_=lambda_, alpha=alpha, sync_period=global_update_period, n_users=n_users, c_mu=c_mu, S=S, R=R, delta=delta, lr=None, alpha_t_scaling=1e-1)

    algorithms['DisLinUCB'] = DisLinUCB(dimension=context_dimension, lambda_=lambda_, delta_=delta, NoiseScale=R, threshold=2, alpha=alpha_linear, alpha_t_scaling=1.0)
    algorithms['One_UCB_GLM'] = One_UCB_GLM(dim=context_dimension, horizon=testing_iterations, numUsers=n_users, lambda_=lambda_, delta=delta, alpha=alpha, alpha_t_scaling=1.0, max_iters=10, R=R)
    algorithms['N_UCB_GLM'] = N_UCB_GLM(dim=context_dimension, horizon=testing_iterations, numUsers=n_users, lambda_=lambda_, c_mu=c_mu, delta=delta, alpha=alpha, alpha_t_scaling=1.0, max_iters=10, R=R)
    algorithms['N_ONS_GLM'] = N_ONS_GLM(dimension=context_dimension, lambda_=lambda_, alpha=alpha, alpha_t_scaling=1.0, c_mu=c_mu, delta=delta, S=S, R=R)

    startTime = datetime.datetime.now()
    realExperiment.runAlgorithms(algorithms, startTime)