#! python3
# usage: bandit.py option 

import os, sys, itertools
import numpy as np
import matplotlib.pyplot as plt

class k_armed_bandit():
    def __init__(self, n_arms, reward_mean_mean, reward_mean_variance, reward_variance, walk_mean, walk_variance):
        self.n_arms = n_arms
        self.means = np.random.normal(loc=reward_mean_mean, scale=np.sqrt(reward_mean_variance), size=n_arms).tolist()
        self.stds = [np.sqrt(reward_variance)]*n_arms
        self.walk_means = [walk_mean]*n_arms
        self.walk_stds = [np.sqrt(walk_variance)]*n_arms
        self.optimal_action = np.argmax(self.means)
        #print("bandit created:")
        #print(self.means, self.optimal_action)

    def reward(self, action):
        r = np.random.normal(self.means[action], self.stds[action])
        self.optimal_action = np.argmax(self.means)
        
        for i in range(self.n_arms):
            walk = np.random.normal(self.walk_means[i], self.walk_stds[i])
            self.means[action] += walk
        
        return r, True if action == self.optimal_action else False

class bandit_testbed():
    def __init__(self, n_bandit, bandit_factory, policy_factory):
        self.bandits = []
        self.policies = []
        for i in range(n_bandit):
            self.bandits.append(bandit_factory())
            self.policies.append(policy_factory())

    def run(self, n_steps):
        n_bandit = len(self.bandits)
        avg_reward = []
        avg_optimal_count = []
        for t in range(n_steps):
            sum_reward = 0.0
            sum_optimal_count = 0.0
            for i in range(n_bandit):
                action = self.policies[i].action()
                reward, hit_optimal = self.bandits[i].reward(action)
                self.policies[i].update(action, reward)
                sum_reward += reward #+= 1.0/(t+1)*(reward-sum_reward[i])
                if hit_optimal:
                    sum_optimal_count += 1.0
            avg_reward.append(sum_reward/n_bandit)
            avg_optimal_count.append(sum_optimal_count/n_bandit)
        return avg_reward, avg_optimal_count

def exploit_greedy(policy):
    return np.argmax(policy.Q)
    
def exploit_UCB(degree_of_exploration):    
    def _exploit_UCB(policy):
        i = np.argmin(policy.N)
        if policy.N[i] == 0:
            return i
        UCB = policy.Q+degree_of_exploration*np.sqrt(np.log(policy.t)/policy.N)
        return np.argmax(UCB)
    return _exploit_UCB
    
def estimate_sample_average(policy, action, reward):
    policy.N[action] += 1
    policy.Q[action] += 1/policy.N[action]*(reward-policy.Q[action])
        
def estimate_const_learning_rate(learning_rate):
    def _estimate_const_learning_rate(policy, action, reward):
        policy.N[action] += 1
        policy.Q[action] += learning_rate*(reward-policy.Q[action])
    return _estimate_const_learning_rate
    
class bandit_policy():
    def __init__(self, n_arms, epsilon, Q1 = 0.0, exploit = exploit_greedy, estimate = estimate_sample_average):
        self.init(n_arms, epsilon, Q1)
        self.exploit = exploit
        self.estimate = estimate
        
    def init(self, n_arms, epsilon, Q1 = 0.0):
        self.Q = np.full(n_arms, Q1)
        self.N = np.full(n_arms, 0.0)
        self.epsilon = epsilon
        self.t = 0
        
    def action(self):
        ticket = np.random.random()
        if ticket < self.epsilon:
            # exploration
            return np.random.randint(0, len(self.Q))
        #exploitation
        return self.exploit(self)
        
    def update(self, action, reward):
        self.estimate(self, action, reward)
        self.t += 1
    
def softmax(a):
    return np.exp(a)/np.sum(np.exp(a))
    
def event_match(a, b):
    return 1 if a == b else 0
    
class bandit_policy_gradient_ascent(bandit_policy):
    def __init__(self, n_arms, step_size, Q1 = 0.0, H1 = 0.0, estimate = estimate_sample_average, no_baseline = False):
        super(bandit_policy_gradient_ascent, self).__init__(n_arms, epsilon = 0.0, Q1 = Q1, exploit = None, estimate = estimate)
        self.step_size = step_size
        self.H = np.full(n_arms, H1)
        self.P = softmax(self.H)
        if no_baseline:
            self.baseline = self.baseline_none
        else:
            self.baseline = self.baseline_average
        
    def action(self):
        return np.random.choice(len(self.H), p=self.P)
        
    def baseline_average(self, i):
        return self.Q[i]
    def baseline_none(self, i):
        return 0
        
    def update(self, action, reward):
        self.estimate(self, action, reward)
        for i in range(len(self.H)):
            self.H[i] += self.step_size*(reward-self.baseline(i))*(event_match(action, i)-self.P[i])
        self.P = softmax(self.H)
        self.t += 1
        
def draw_result(reward_list, optimal_count_list, labels):
    plt.figure(1)
    
    plt.subplot(211)
    for i in range(len(reward_list)):
        plt.plot(reward_list[i], label=labels[i])
    plt.xlabel('Steps')
    plt.ylabel('average reward')
    plt.legend()
        
    plt.subplot(212)
    for i in range(len(optimal_count_list)):
        plt.plot(optimal_count_list[i], label=labels[i])
    plt.xlabel('Steps')
    plt.ylabel('optimal action ratio')
    plt.legend()
    
    plt.show()

# Fig 2.2
def compare_greedy():
    n_bandit = 2000
    reward_list = []
    optimal_count_list = []
    labels = ["e=0", "e=0.01", "e=0.1"]
    bandit_factory = lambda: k_armed_bandit(n_arms=10, reward_mean_mean=0.0, reward_mean_variance=1.0, reward_variance=1.0, walk_mean=0.0, walk_variance=0.0)
    for epsilon in [0.0, 0.01, 0.1]: 
        policy_factory = lambda: bandit_policy(10, epsilon)
        testbed = bandit_testbed(n_bandit, bandit_factory, policy_factory)
        R, optimal_count = testbed.run(1000)
        reward_list.append(R)
        optimal_count_list.append(optimal_count)
    draw_result(reward_list, optimal_count_list, labels)
        
# Ex 2.5
def compare_nonstationary():
    n_bandit = 2000
    epsilon = 0.1
    reward_list = []
    optimal_count_list = []
    labels = ["sample_average", "const_learning_rate", "UCB c=2"]
    bandit_factory = lambda: k_armed_bandit(n_arms=10, reward_mean_mean=0.0, reward_mean_variance=1.0, reward_variance=1.0, walk_mean=0.0, walk_variance=0.0001)
    for policy_factory in [lambda: bandit_policy(10, epsilon, 0.0, exploit_greedy, estimate_sample_average), lambda: bandit_policy(10, epsilon, 0.0, exploit_greedy, estimate_const_learning_rate(0.1)), lambda: bandit_policy(10, epsilon=0.0, Q1 = 0.0, exploit=exploit_UCB(2.0))]:
        testbed = bandit_testbed(n_bandit, bandit_factory, policy_factory)
        R, optimal_count = testbed.run(10000)
        reward_list.append(R)
        optimal_count_list.append(optimal_count)
    draw_result(reward_list, optimal_count_list, labels)
   
# Fig 2.3   
def compare_optimistic_initial():
    n_bandit = 2000
    reward_list = []
    optimal_count_list = []
    labels = ["epsilon = 0.1, Q1 = 0.0", "epsilon = 0, Q1 = 5", "epsilon = 0, Q1 = 20"]
    bandit_factory = lambda: k_armed_bandit(n_arms=10, reward_mean_mean=0.0, reward_mean_variance=1.0, reward_variance=1.0, walk_mean=0.0, walk_variance=0.0)
    for policy_factory in [lambda: bandit_policy(10, epsilon=0.1, Q1 = 0.0, estimate = estimate_const_learning_rate(0.1)), lambda: bandit_policy(10, epsilon=0.0, Q1 = 5.0, estimate = estimate_const_learning_rate(0.1)), lambda: bandit_policy(10, epsilon=0.0, Q1 = 20.0, estimate = estimate_const_learning_rate(0.1)) ]: 
        testbed = bandit_testbed(n_bandit, bandit_factory, policy_factory)
        R, optimal_count = testbed.run(1000)
        reward_list.append(R)
        optimal_count_list.append(optimal_count)
    draw_result(reward_list, optimal_count_list, labels)
    
# Fig 2.4
def compare_UCB():
    n_bandit = 2000
    reward_list = []
    optimal_count_list = []
    labels = ["greedy e=0.1", "UCB c=2"]
    bandit_factory = lambda: k_armed_bandit(n_arms=10, reward_mean_mean=0.0, reward_mean_variance=1.0, reward_variance=1.0, walk_mean=0.0, walk_variance=0.0)
    for policy_factory in [lambda: bandit_policy(10, epsilon=0.1, Q1 = 0.0, exploit=exploit_greedy), lambda: bandit_policy(10, epsilon=0.0, Q1 = 0.0, exploit=exploit_UCB(2.0)) ]: 
        testbed = bandit_testbed(n_bandit, bandit_factory, policy_factory)
        R, optimal_count = testbed.run(1000)
        reward_list.append(R)
        optimal_count_list.append(optimal_count)
    draw_result(reward_list, optimal_count_list, labels)
    
# Fig 2.5
def test_gradient():
    n_bandit = 2000
    reward_list = []
    optimal_count_list = []
    labels = ["step_size = 0.1 with baseline", "without baseline"]
    bandit_factory = lambda: k_armed_bandit(n_arms=10, reward_mean_mean=0.0, reward_mean_variance=1.0, reward_variance=1.0, walk_mean=0.0, walk_variance=0.0)
    for policy_factory in [ lambda: bandit_policy_gradient_ascent(10, 0.1), lambda: bandit_policy_gradient_ascent(10, 0.1, no_baseline = True) ]: 
        testbed = bandit_testbed(n_bandit, bandit_factory, policy_factory)
        R, optimal_count = testbed.run(1000)
        reward_list.append(R)
        optimal_count_list.append(optimal_count)
    draw_result(reward_list, optimal_count_list, labels)
        
if __name__ == "__main__":
    globals()[sys.argv[1]]()