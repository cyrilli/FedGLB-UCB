import numpy as np
import copy


"""
FedGLB_UCB variant 1

scheduled communication
no local update

"""

def sigmoid(x):
    return 1/(1+np.exp(-x))

class LocalClient:
    def __init__(self, featureDimension, lambda_, c_mu, S, R, delta):
        self.d = featureDimension
        self.lambda_ = lambda_
        self.c_mu = c_mu
        self.R = R 
        self.S = S
        self.delta = delta
        # Sufficient statistics stored on the client #
        # latest local sufficient statistics
        self.A = np.zeros((self.d, self.d))
        self.X = np.zeros((0, self.d))
        self.y = np.zeros((0,))
        self.A_uploadbuffer = np.zeros((self.d, self.d))

        # for computing UCB
        self.AInv = self.c_mu/self.lambda_ * np.identity(self.d)
        self.Theta = np.zeros(self.d)  # center of confidence ellipsoid

        self.alpha_t = np.sqrt(self.lambda_/self.c_mu*self.S**2)

    def localUpdate(self, articlePicked_FeatureVector, click):
        self.A_uploadbuffer += np.outer(articlePicked_FeatureVector, articlePicked_FeatureVector)
        self.X = np.concatenate((self.X, articlePicked_FeatureVector.reshape(1, self.d)), axis=0)
        self.y = np.concatenate((self.y, np.array([click])), axis=0)


class FedGLB_UCB_v1:
    def __init__(self, dimension, lambda_, sync_period, delta, alpha = None, alpha_t_scaling=1.0, R=0.5, S=1, c_mu=0.25, max_iters=None, init_x =None):
        self.dimension = dimension
        self.lambda_ = lambda_
        self.c_mu = c_mu
        self.R = R
        self.S = S
        self.delta = delta
        self.CanEstimateUserPreference = False  # set to true if want to record parameter estimation error
        self.clients = {}
        # aggregated sufficient statistics of all clients
        self.A_g = self.lambda_/self.c_mu * np.identity(n=self.dimension)

        self.init_x = init_x
        self.syncPeriod = round(sync_period)
        self.numObs = 0  # total number of observations across all clients
        self.GlobalTheta = np.zeros(self.dimension)
        self.lastGlobalTheta = np.zeros(self.dimension)

        self.max_iters = max_iters
        self.alpha = alpha
        self.alpha_t_scaling = alpha_t_scaling

        # records
        self.totalCommCost = 0

    def decide(self, arm_matrix, currentClientID):
        if currentClientID not in self.clients:
            self.clients[currentClientID] = LocalClient(self.dimension, self.lambda_, self.c_mu, self.S, self.R, self.delta)

        ucbs = np.sqrt((np.matmul(arm_matrix, self.clients[currentClientID].AInv) * arm_matrix).sum(axis=1))

        if self.alpha is None:
            alpha_t = self.clients[currentClientID].alpha_t * self.alpha_t_scaling
        else:
            alpha_t = self.alpha
        # Compute UCB
        mu = np.matmul(arm_matrix, self.clients[currentClientID].Theta) + alpha_t * ucbs
        # Argmax breaking ties randomly
        arm = np.random.choice(np.flatnonzero(mu == mu.max()))

        return arm_matrix[arm], arm

    def updateParameters(self, articlePickedVec, click, currentClientID):
        self.numObs += 1
        # update upload buffer, but not local bandit model
        self.clients[currentClientID].localUpdate(articlePickedVec, click)

        if self.numObs % self.syncPeriod == 0:
            # first collect the local updates from all the clients
            for clientID, clientModel in self.clients.items():
                if clientID != currentClientID:
                    # self.totalCommCost += 1
                    self.totalCommCost += (self.dimension*self.dimension + 1)
                self.A_g += clientModel.A_uploadbuffer
                clientModel.A_uploadbuffer = np.zeros((self.dimension, self.dimension))
            # run J steps of AGD with all the data collected
            if self.init_x is None:
                x = self.lastGlobalTheta
            else:
                x = self.init_x

            lambda_prev = 0
            lambda_curr = 1
            gamma = 1
            y_prev = x
            step_size = 1/(0.25+self.lambda_/self.numObs)
            if self.max_iters is None:
                max_iters = self.numObs*2
            else:
                max_iters = self.max_iters
            for iter in range(max_iters):
                # collect and aggregate local gradients w.r.t. model x
                gradient = np.zeros(self.dimension)
                for clientID, clientModel in self.clients.items():
                    z = np.dot(clientModel.X, x)
                    gradient += np.dot(np.transpose(clientModel.X), -clientModel.y + sigmoid(z))
                gradient += self.lambda_ * x
                gradient = gradient / self.numObs

                if np.linalg.norm(gradient) <= np.sqrt((2*self.lambda_)/self.numObs**3):
                    break

                # one step of AGD update
                y_curr = x - step_size * gradient
                x = (1 - gamma) * y_curr + gamma * y_prev
                y_prev = y_curr

                lambda_tmp = lambda_curr
                lambda_curr = (1 + np.sqrt(1 + 4 * lambda_prev * lambda_prev)) / 2
                lambda_prev = lambda_tmp
                gamma = (1 - lambda_prev) / lambda_curr

            # self.totalCommCost += iter*(len(self.clients)-1)*2
            self.totalCommCost += iter*(len(self.clients)-1)*2*self.dimension

            alpha_t =  1 / self.c_mu * np.sqrt(np.dot(np.dot(gradient*self.numObs, self.A_g), gradient*self.numObs)) \
                + self.R / self.c_mu * np.sqrt(self.dimension * np.log(1+self.numObs*self.c_mu/(self.dimension*self.lambda_))+2*np.log(1.0/self.delta)) + np.sqrt(self.lambda_/self.c_mu)*self.S

            self.GlobalTheta = copy.deepcopy(x)
            self.lastGlobalTheta = copy.deepcopy(x)
            AInv_g = np.linalg.inv(self.A_g)

            for clientID, clientModel in self.clients.items():
                if clientID != currentClientID:
                    # self.totalCommCost += 1
                    self.totalCommCost += (self.dimension*self.dimension + self.dimension + 1)
                clientModel.A = copy.deepcopy(self.A_g)
                clientModel.AInv = copy.deepcopy(AInv_g)
                clientModel.Theta = copy.deepcopy(self.GlobalTheta)
                clientModel.alpha_t = alpha_t

    def getTheta(self, clientID):
        return self.clients[clientID].Theta


