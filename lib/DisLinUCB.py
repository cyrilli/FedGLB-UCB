import numpy as np
import copy
import random

class LocalClient:
    def __init__(self, featureDimension, lambda_, delta_, NoiseScale):
        self.d = featureDimension
        self.lambda_ = lambda_
        self.delta_ = delta_
        self.NoiseScale = NoiseScale

        # Sufficient statistics stored on the client #
        # latest local sufficient statistics
        self.A_local = np.zeros((self.d, self.d))  #lambda_ * np.identity(n=self.d)
        self.b_local = np.zeros(self.d)
        self.numObs_local = 0

        # aggregated sufficient statistics recently downloaded
        self.A_uploadbuffer = np.zeros((self.d, self.d))
        self.b_uploadbuffer = np.zeros(self.d)
        self.numObs_uploadbuffer = 0

        # for computing UCB
        self.AInv = np.linalg.inv(self.A_local+self.lambda_ * np.identity(n=self.d))
        self.UserTheta = np.zeros(self.d)

        self.alpha_t = self.NoiseScale * np.sqrt(
            self.d * np.log(1 + (self.numObs_local) / (self.d * self.lambda_)) + 2 * np.log(1 / self.delta_)) + np.sqrt(
            self.lambda_)


    def getUCB(self, alpha, article_FeatureVector):
        if alpha == -1:
            alpha = self.alpha_t

        mean = np.dot(self.UserTheta, article_FeatureVector)
        var = np.sqrt(np.dot(np.dot(article_FeatureVector, self.AInv), article_FeatureVector))
        pta = mean + alpha * var
        return pta

    def localUpdate(self, articlePicked_FeatureVector, click):
        self.A_local += np.outer(articlePicked_FeatureVector, articlePicked_FeatureVector)
        self.b_local += articlePicked_FeatureVector * click
        self.numObs_local += 1

        self.A_uploadbuffer += np.outer(articlePicked_FeatureVector, articlePicked_FeatureVector)
        self.b_uploadbuffer += articlePicked_FeatureVector * click
        self.numObs_uploadbuffer += 1

        self.AInv = np.linalg.inv(self.A_local+self.lambda_ * np.identity(n=self.d))
        self.UserTheta = np.dot(self.AInv, self.b_local)

        self.alpha_t = self.NoiseScale * np.sqrt(
            self.d * np.log(1 + (self.numObs_local) / (self.d * self.lambda_)) + 2 * np.log(1 / self.delta_)) + np.sqrt(
            self.lambda_)

    def getTheta(self):
        return self.UserTheta

    def syncRoundTriggered(self, threshold):
        numerator = np.linalg.det(self.A_local+self.lambda_ * np.identity(n=self.d))
        denominator = np.linalg.det(self.A_local-self.A_uploadbuffer+self.lambda_ * np.identity(n=self.d))
        return np.log(numerator/denominator)*(self.numObs_uploadbuffer) >= threshold

class DisLinUCB:
    def __init__(self, dimension, lambda_, delta_, NoiseScale, threshold, alpha=None, alpha_t_scaling=1.0):
        self.dimension = dimension
        self.alpha = alpha
        self.alpha_t_scaling = alpha_t_scaling
        self.lambda_ = lambda_
        self.delta_ = delta_
        self.NoiseScale = NoiseScale
        self.threshold = threshold
        self.CanEstimateUserPreference = False

        self.clients = {}
        # aggregated sufficient statistics of all clients
        self.A_aggregated = np.zeros((self.dimension, self.dimension))
        self.b_aggregated = np.zeros(self.dimension)
        self.numObs_aggregated = 0


        # records
        self.totalCommCost = 0

    def decide(self, arm_matrix, clientID):
        # start = time.time()
        if clientID not in self.clients:
            self.clients[clientID] = LocalClient(self.dimension, self.lambda_, self.delta_, self.NoiseScale)

        ucbs = np.sqrt((np.matmul(arm_matrix, self.clients[clientID].AInv) * arm_matrix).sum(axis=1))

        if self.alpha is None:
            alpha_t = self.clients[clientID].alpha_t * self.alpha_t_scaling
        else:
            alpha_t = self.alpha
        # Compute UCB
        mu = np.matmul(arm_matrix, self.clients[clientID].UserTheta) + alpha_t * ucbs
        # Argmax breaking ties randomly
        arm = np.random.choice(np.flatnonzero(mu == mu.max()))
        # end = time.time()
        # print("v0 select takes: {}".format(end - start))
        return arm_matrix[arm], arm

    def updateParameters(self, articlePickedVec, click, currentClientID):
        # update local ss, and upload buffer
        self.clients[currentClientID].localUpdate(articlePickedVec, click)

        if self.clients[currentClientID].syncRoundTriggered(self.threshold):
            # a round of global synchronization is triggered
            # first collect the local updates of all the clients
            for clientID, clientModel in self.clients.items():
                # self.totalCommCost += 1
                self.totalCommCost += (self.dimension**2 + self.dimension + 1)
                self.A_aggregated += clientModel.A_uploadbuffer
                self.b_aggregated += clientModel.b_uploadbuffer
                self.numObs_aggregated += clientModel.numObs_uploadbuffer

                # clear client's upload buffer
                clientModel.A_uploadbuffer = np.zeros((self.dimension, self.dimension))
                clientModel.b_uploadbuffer = np.zeros(self.dimension)
                clientModel.numObs_uploadbuffer = 0
            # then send the aggregated ss to all the clients, now all of them are synced
            for clientID, clientModel in self.clients.items():
                # self.totalCommCost += 1
                self.totalCommCost += (self.dimension**2 + self.dimension + 1)
                clientModel.A_local = copy.deepcopy(self.A_aggregated)
                clientModel.b_local = copy.deepcopy(self.b_aggregated)
                clientModel.numObs_local = copy.deepcopy(self.numObs_aggregated)
    def getTheta(self, clientID):
        return self.clients[clientID].UserTheta


