import numpy as np
import copy
import time
from cvxopt import matrix, solvers

"""
FedGLB_UCB variant 3

scheduled communication
ONS for both global and local update

"""

def sigmoid(x):
    return 1/(1+np.exp(-x))

def projection_in_norm(x, M):
    """Projection of x to simplex indiced by matrix M. Uses quadratic programming."""
    x = np.mat(x).T
    M = np.mat(M)
    m = M.shape[0]

    P = matrix(2 * M)
    q = matrix(-2 * M * x)
    G = matrix(-np.eye(m))
    h = matrix(np.zeros((m, 1)))
    A = matrix(np.ones((1, m)))
    b = matrix(1.0)

    sol = solvers.qp(P, q, G, h, A, b)
    solvers.options['show_progress'] = False
    return np.squeeze(sol["x"])

class LocalClient:
    def __init__(self, featureDimension, lambda_, n_users, c_mu, gamma, delta, S, R):
        self.d = featureDimension
        self.lambda_ = lambda_
        self.delta = delta
        self.c_mu = c_mu
        self.gamma = gamma
        self.S = S
        self.R = R
        # Sufficient statistics stored on the client #
        # latest local sufficient statistics
        self.A = self.lambda_ * np.identity(n=self.d)
        self.V = self.lambda_ * np.identity(n=self.d)
        self.b = np.zeros(self.d)
        self.numObs_local = 0
        self.n_users = n_users
        # aggregated sufficient statistics recently downloaded
        self.V_uploadbuffer = np.zeros((self.d, self.d))
        self.numObs_uploadbuffer = 0
        self.DeltaX = np.zeros((0, self.d))
        self.Deltay = np.zeros((0,))

        # for computing UCB
        self.VInv = 1 / self.lambda_ * np.identity(self.d)
        self.ThetaRidge = np.zeros(self.d)  # center of confidence ellipsoid
        self.ThetaONS = np.zeros(self.d)  # ONS estimation

        self.Bt = 2 * self.gamma * self.lambda_ * self.S**2
        self.beta_t = self.lambda_*self.S**2
        self.sum_z_sqr = 0

    def localUpdate(self, articlePicked_FeatureVector, click):
        self.numObs_local += 1

        z = articlePicked_FeatureVector.dot(self.ThetaONS)
        grad = (-click + sigmoid(z)) * articlePicked_FeatureVector
        self.A += np.outer(grad, grad)
        self.V += np.outer(articlePicked_FeatureVector, articlePicked_FeatureVector)
        self.b += z * articlePicked_FeatureVector
        tmp_vec = self.VInv.dot(articlePicked_FeatureVector)
        self.VInv = self.VInv - np.outer(tmp_vec, tmp_vec) / (1 + articlePicked_FeatureVector.dot(tmp_vec))  # sherman-morrison formula

        self.V_uploadbuffer += np.outer(articlePicked_FeatureVector, articlePicked_FeatureVector)
        self.DeltaX = np.concatenate((self.DeltaX, articlePicked_FeatureVector.reshape(1, self.d)), axis=0)
        self.Deltay = np.concatenate((self.Deltay, np.array([click])), axis=0)
        self.numObs_uploadbuffer += 1

        # run ONS update step with the new data point
        AInv = np.linalg.inv(self.A)
        theta_prime = self.ThetaONS - 1 / self.gamma * AInv.dot(grad)
        self.ThetaONS = projection_in_norm(theta_prime, self.A)

        self.ThetaRidge = np.dot(self.VInv, self.b)

        self.Bt += 1/(2*self.gamma) * np.dot(np.dot(grad, AInv), grad)
        self.beta_t = self.lambda_*self.S**2 + 1 + 4/self.c_mu*self.Bt + 8*self.R**2/self.c_mu**2*np.log(self.n_users/self.delta*np.sqrt(4+8/self.c_mu*self.Bt+64*self.R**2/(4*self.delta**2*self.c_mu**4)))
        self.sum_z_sqr += z**2

    def syncRoundTriggered(self, threshold):
        numerator = np.linalg.det(self.V)
        denominator = np.linalg.det(self.V-self.V_uploadbuffer)
        return np.log(numerator/denominator)*self.numObs_uploadbuffer >= threshold

class FedGLB_UCB_v3:
    def __init__(self, dimension, lambda_, alpha, sync_period, n_users, c_mu=0.25, S=1, R=0.5, delta=1e-2, lr=None, alpha_t_scaling=1.0):
        self.dimension = dimension
        self.lambda_ = lambda_
        self.c_mu = c_mu
        self.S = S
        self.R = R
        self.delta = delta
        self.sync_period = round(sync_period)
        self.CanEstimateUserPreference = False  # set to true if want to record parameter estimation error
        self.alpha = alpha
        self.alpha_t_scaling = alpha_t_scaling
        self.clients = {}
        # aggregated sufficient statistics of all clients
        self.A_g = self.lambda_ * np.identity(n=self.dimension)
        self.V_g = self.lambda_ * np.identity(n=self.dimension)
        self.b_g = np.zeros(self.dimension)
        self.n_users = n_users
        self.numObs_g = 0
        self.numObs = 0
        self.ThetaONS = np.zeros(self.dimension)  # the ons estimate on server is only updated at each global update
        self.lr = lr

        self.gamma = 0.5 * min(1 / (4 * self.S * np.sqrt(0.25**2 * self.S**2 + self.R**2)), self.c_mu / ((0.25**2 * self.S**2 + self.R**2)*self.sync_period))
        self.Bt_g = 2 * self.gamma * self.lambda_ * self.S**2
        self.sum_z_sqr_g = 0

        # records
        self.totalCommCost = 0
        self.syncRound = False

    def decide(self, arm_matrix, currentClientID):
        if currentClientID not in self.clients:
            self.clients[currentClientID] = LocalClient(self.dimension, self.lambda_, self.n_users, self.c_mu, self.delta, self.gamma, self.S, self.R)
        ucbs = np.sqrt((np.matmul(arm_matrix, self.clients[currentClientID].VInv) * arm_matrix).sum(axis=1))


        if self.alpha is not None:
            alpha_t = self.alpha
        else:
            alpha_t = np.sqrt(self.clients[currentClientID].beta_t-self.clients[currentClientID].sum_z_sqr+np.dot(self.clients[currentClientID].ThetaRidge,self.clients[currentClientID].b))
            alpha_t = self.alpha_t_scaling * alpha_t
        # Compute UCB
        mu = np.matmul(arm_matrix, self.clients[currentClientID].ThetaRidge) + alpha_t * ucbs
        # Argmax breaking ties randomly
        arm = np.random.choice(np.flatnonzero(mu == mu.max()))
        return arm_matrix[arm], arm

    def updateParameters(self, articlePickedVec, click, currentClientID):
        self.numObs += 1
        # update local ss, and upload buffer
        self.clients[currentClientID].localUpdate(articlePickedVec, click)

        if self.numObs % self.sync_period == 0:
            # first collect the local updates from all the clients
            grad = np.zeros(self.dimension)  # gradient w.r.t. last synced ThetaONS
            numObs_epoch = 0
            for clientID, clientModel in self.clients.items():
                # self.totalCommCost += 1
                self.totalCommCost += (self.dimension*self.dimension + self.dimension + 1)
                self.V_g += clientModel.V_uploadbuffer
                self.b_g += np.dot(clientModel.V_uploadbuffer, self.ThetaONS)  # sum x x^T theta, using last synced ThetaONS

                z = np.dot(clientModel.DeltaX, self.ThetaONS)
                self.sum_z_sqr_g += np.linalg.norm(z)**2
                grad += np.dot(np.transpose(clientModel.DeltaX), -clientModel.Deltay+sigmoid(z))

                clientModel.DeltaX = np.zeros((0, self.dimension))
                clientModel.Deltay = np.zeros((0,))
                self.numObs_g += clientModel.numObs_uploadbuffer
                numObs_epoch += clientModel.numObs_uploadbuffer
                clientModel.numObs_uploadbuffer = 0
                clientModel.V_uploadbuffer = np.zeros((self.dimension, self.dimension))
            assert self.numObs_g == self.numObs
            self.A_g += np.outer(grad, grad)
            AInv_g = np.linalg.inv(self.A_g)
            # run one step of ONS with all the data collected in this epoch
            if self.lr is None:
                theta_prime = self.ThetaONS - 1.0/self.gamma * AInv_g.dot(grad)
            else:
                theta_prime = self.ThetaONS - 1.0/self.lr * AInv_g.dot(grad)
            self.ThetaONS = projection_in_norm(theta_prime, self.A_g)
            VInv_g = np.linalg.inv(self.V_g)
            ThetaRidge = np.dot(VInv_g, self.b_g)

            self.Bt_g += 1/(2*self.gamma) * np.dot(np.dot(grad, AInv_g), grad)
            beta_g = self.lambda_*self.S**2 + 1 + 4/self.c_mu*self.Bt_g + 8*self.R**2/self.c_mu**2 *np.log(self.n_users/self.delta*np.sqrt(4+8/self.c_mu*self.Bt_g+64*self.R**2/(4*self.delta**2*self.c_mu**4)))

            for clientID, clientModel in self.clients.items():
                # self.totalCommCost += 1
                self.totalCommCost += (self.dimension * self.dimension * 2 + self.dimension + 1)
                clientModel.A = copy.deepcopy(self.A_g)
                clientModel.b = copy.deepcopy(self.b_g)
                clientModel.V = copy.deepcopy(self.V_g)
                clientModel.VInv = copy.deepcopy(VInv_g) #np.linalg.inv(clientModel.A_local + self.lambda_ * np.identity(n=self.dimension))
                clientModel.numObs_local = copy.deepcopy(self.numObs_g)

                clientModel.ThetaRidge = copy.deepcopy(ThetaRidge)  # center of confidence ellipsoid
                clientModel.ThetaONS = copy.deepcopy(self.ThetaONS)  # ONS estimation

                clientModel.Bt = copy.deepcopy(self.Bt_g)
                clientModel.beta_t = copy.deepcopy(beta_g)
                clientModel.sum_z_sqr = copy.deepcopy(self.sum_z_sqr_g)
        # end = time.time()
        # print("v3 update takes: {}".format(end - start))

    def getTheta(self, clientID):
        return self.clients[clientID].ThetaRidge


