import numpy as np
from scipy import special
from sklearn.linear_model import LogisticRegression

class UserStruct:
  def __init__(self, userID, dim):
    self.userID = userID
    self._rewards = np.array([])  # stores all rewards
    self._arms = np.ndarray([0, dim])  # stores arm differences
    self._outer = np.zeros([dim, dim])
    self.UserTheta = np.zeros(dim)

class N_UCB_GLM: 
  """UCB-GLM algorithm.
  See "Provably Optimal Algorithms for Generalized Linear Contextual Bandits",
  by Li et al. (2017).
  """

  def __init__(self, dim, horizon, delta, numUsers, alpha=None, alpha_t_scaling=1.0, lambda_=1., R=0.5, c_mu=0.25, max_iters=100):
    self.lambda_ = lambda_
    self.numUsers = numUsers
    self._dim = dim
    self.alpha = alpha
    self.alpha_t_scaling = alpha_t_scaling
    self.max_iters = max_iters
    # Set confidence interval scaling, by
    # Theorem 2 in Li (2017)
    # Provably Optimal Algorithms for Generalized Linear Contextual Bandits
    delta = delta
    # Confidence ellipsoid width:
    self._ci_scaling = (R / c_mu) * (np.sqrt((self._dim / 2) *
                                     np.log(1. + 2. * horizon / self._dim) +
                                     np.log(1 / delta)))

    self.users = {}

    self.CanEstimateUserPreference = False  # set to true if want to record parameter estimation error
    self.totalCommCost = 0

  def decide(self, arm_matrix, userID):
    """Computes which arm to pull next.
    Args:
      arms: a list of feature vectors, one for each arm
    Returns:
      The selected arm, its index in arms, and the computed scores
    """
    if userID not in self.users:
      self.users[userID] = UserStruct(userID, self._dim)

    gram = self.users[userID]._outer + np.eye(self._dim) * self.lambda_
    gram_inv = np.linalg.inv(gram)
    ucbs = np.sqrt((np.matmul(arm_matrix, gram_inv) * arm_matrix).sum(axis=1))

    if self.alpha is None:
        alpha_t = self._ci_scaling * self.alpha_t_scaling
    else:
        alpha_t = self.alpha

    # Compute UCB
    mu = np.matmul(arm_matrix, self.users[userID].UserTheta) + alpha_t * ucbs
    # Argmax breaking ties randomly
    arm = np.random.choice(np.flatnonzero(mu == mu.max()))

    return arm_matrix[arm], arm

  def updateParameters(self, arm, reward, userID):
    """Updates state with arm and reward.
    Args:
      reward: the reward received
      arm: the arm that was pulled
    """
    assert len(arm) == self._dim, 'Expected dimension {}, got {}'.format(
        self._dim, len(arm))
    self.users[userID]._rewards = np.append(self.users[userID]._rewards, reward)
    self.users[userID]._arms = np.concatenate((self.users[userID]._arms, [arm]), axis=0)
    self.users[userID]._outer += np.outer(arm, arm)

    # Estimate w
    w, _ = self.solve_logistic_bandit_v1(userID, init_iters=0, max_iters=self.max_iters, tol=1e-3)
    # w, _ = self.solve_logistic_bandit_v2(userID, init_iters=0, max_iters=self.max_iters, tol=1e-4)
    self.users[userID].UserTheta = w

  def solve_logistic_bandit_v1(self, userID, init_iters=10, max_iters=20, tol=1e-3):
    """Solves the maximum-likelihood problem.
    Implements iterative reweighted least squares for Bayesian logistic
    regression. See sections 4.3.3 and 4.5.1 in Pattern Recognition and Machine
    Learning, Bishop (2006)
    Args:
      init_iters: number of initial iterations to skip (returns zeros)
      max_iters: number of least squares iterations
      tol: tolerance level of change in solution between iterations before
        terminating
    Returns:
      w: maximum likelihood solution
      gram: Gram matrix
    """

    arms = self.users[userID]._arms
    w = np.zeros(self._dim)
    gram = np.eye(self._dim) * self.lambda_
    if len(self.users[userID]._arms) > init_iters:
      for iter_ in range(max_iters):
        prev_w = np.copy(w)
        arms_w = arms.dot(w)
        sig_arms_w = special.expit(arms_w)
        r = np.diag(sig_arms_w * (1 - sig_arms_w))
        gram = (((arms.T).dot(r)).dot(arms) +
                np.eye(self._dim) * self.lambda_)
        rz = r.dot(arms_w) - (sig_arms_w - self.users[userID]._rewards)
        w = np.linalg.solve(gram, (arms.T).dot(rz))
        if np.linalg.norm(w - prev_w) < tol:
          break
    else:
      iter_ = 0
    return w, iter_

  def solve_logistic_bandit_v2(self, userID, init_iters=10, max_iters=20, tol=1e-3):
    w = np.zeros(self._dim)
    if (len(self.users[userID]._arms) > init_iters) and (len(np.unique(self.users[userID]._rewards)) > 1):
        clf = LogisticRegression(penalty='none', fit_intercept=False, solver='lbfgs', max_iter=max_iters, tol=tol).fit(self.users[userID]._arms, self.users[userID]._rewards)
        w = clf.coef_[0]
        iter_ = clf.n_iter_[0]
    else:
      iter_ = 0
    return w, iter_

  def getTheta(self, userID):
    return self.users[userID].UserTheta
