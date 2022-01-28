import numpy as np
import copy
from cvxopt import matrix, solvers
solvers.options['show_progress'] = False
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
    return np.squeeze(sol["x"])

class LocalClient:
    def __init__(self, featureDimension, lambda_, c_mu, delta, S, R):
        self.d = featureDimension
        self.lambda_ = lambda_
        self.delta = delta
        self.S = S
        self.c_mu = c_mu
        self.R = R  # because reward is binary / bounded in [0,1]

        # Sufficient statistics stored on the client #
        # latest local sufficient statistics
        self.A = np.zeros((self.d, self.d))
        self.b = np.zeros(self.d)
        self.numObs_local = 0

        # for computing UCB
        # self.VInv = 1/self.lambda_ * np.identity(self.d)
        self.AInv = 1/self.lambda_ * np.identity(self.d)
        self.ThetaRidge = np.zeros(self.d)  # center of confidence ellipsoid
        self.ThetaONS = np.zeros(self.d)  # ONS estimation

        self.Bt = 2*self.c_mu*self.lambda_*self.S**2

        self.sum_z_sqr = 0
        self.alpha_t = np.sqrt(
            self.lambda_*self.S**2 + 1 + 4 / self.c_mu * self.Bt + (8 * self.R ** 2) / self.c_mu ** 2 * np.log(
                2 / self.delta * np.sqrt(1 + 2 / self.c_mu * self.Bt + (4 * self.R ** 4) / (
                        self.c_mu ** 4 * self.delta ** 2))) + np.dot(self.ThetaRidge, self.b) - self.sum_z_sqr)

    def localUpdate(self, articlePicked_FeatureVector, click):
        self.A += np.outer(articlePicked_FeatureVector, articlePicked_FeatureVector)
        self.numObs_local += 1

        # get predicted reward using ThetaONS
        z = articlePicked_FeatureVector.dot(self.ThetaONS)
        self.b += z * articlePicked_FeatureVector
        self.sum_z_sqr += z**2

        tmp = self.AInv.dot(articlePicked_FeatureVector)
        self.AInv -= np.outer(tmp, tmp) / (1 + articlePicked_FeatureVector.dot(tmp))  # sherman-morrison formula

        # update ONS with the new data point
        grad = -click + sigmoid(np.dot(articlePicked_FeatureVector, self.ThetaONS))
        
        theta_prime = self.ThetaONS - grad / self.c_mu * self.AInv.dot(articlePicked_FeatureVector)
        self.ThetaONS = projection_in_norm(theta_prime, self.A+self.lambda_/self.c_mu * np.identity(n=self.d))
        self.ThetaRidge = np.dot(self.AInv, self.b)

        self.Bt += 1/(2*self.c_mu) * (grad**2) * np.dot(np.dot(articlePicked_FeatureVector, self.AInv), articlePicked_FeatureVector)
        self.alpha_t = np.sqrt(
            self.lambda_*self.S**2 + 1 + 4 / self.c_mu * self.Bt + (8 * self.R ** 2) / self.c_mu ** 2 * np.log(
                2 / self.delta * np.sqrt(1 + 2 / self.c_mu * self.Bt + (4 * self.R ** 4) / (
                        self.c_mu ** 4 * self.delta ** 2))) + np.dot(self.ThetaRidge, self.b) - self.sum_z_sqr)


class N_ONS_GLM:
    def __init__(self, dimension, lambda_, alpha=None, alpha_t_scaling=1.0, c_mu=0.25, delta=1e-2, S=1, R=0.5):
        self.dimension = dimension
        self.alpha = alpha
        self.alpha_t_scaling = alpha_t_scaling
        self.lambda_ = lambda_
        self.c_mu = c_mu 
        self.delta = delta
        self.S = S
        self.R = R
        self.CanEstimateUserPreference = False  # set to true if want to record parameter estimation error
        self.clients = {}

        # records
        self.totalCommCost = 0

    def decide(self, arm_matrix, currentClientID):
        if currentClientID not in self.clients:
            self.clients[currentClientID] = LocalClient(self.dimension, self.lambda_, self.c_mu, self.delta, self.S, self.R)

        ucbs = np.sqrt((np.matmul(arm_matrix, self.clients[currentClientID].AInv) * arm_matrix).sum(axis=1))

        if self.alpha is None:
            alpha_t = self.clients[currentClientID].alpha_t * self.alpha_t_scaling
        else:
            alpha_t = self.alpha

        # Compute UCB
        mu = np.matmul(arm_matrix, self.clients[currentClientID].ThetaRidge) + alpha_t * ucbs
        # Argmax breaking ties randomly
        arm = np.random.choice(np.flatnonzero(mu == mu.max()))

        return arm_matrix[arm], arm

    def updateParameters(self, articlePickedVec, click, currentClientID):
        # update local ss, and upload buffer
        self.clients[currentClientID].localUpdate(articlePickedVec, click)

    def getTheta(self, clientID):
        return self.clients[clientID].ThetaRidge


