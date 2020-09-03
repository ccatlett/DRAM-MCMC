""" Knapsack DR MCMC w/ Plots
    Author: C. Catlett
    Date:   July 15, 2020
    Desc:   Delayed Rejection MCMC routine with simulated annealing to optimize Knapsack Problem
            Altered from tutorial found at https://mlwhiz.com/blog/2015/08/21/mcmc_algorithm_cryptography/"""

import numpy as np
import matplotlib as mb
import random, sys

weights = [20,40,60,12,34,45,67,33,23,12,34,56,23,56]
gold = [120,420,610,112,341,435,657,363,273,812,534,356,223,516]
weight_max = 400
num_boxes = len(weights)

# Given the current theta, propose new candidate according to weight
# constraint by picking up/putting down one box
def proposal(theta_curr):
    # Select and pick up/put down one box
    ind = random.randint(0,num_boxes-1)
    theta_prop = list(theta_curr)
    theta_prop[ind] = 1 - theta_curr[ind]

    # Change proposal if over weight limit
    print(np.dot(theta_prop, weights))
    if np.dot(theta_prop, weights) > weight_max:
      print("reject")
      proposal(theta_prop)
    return theta_prop

# Returns log of objective function: e^(beta * gold picked up)
def eval_theta(beta, theta):
  return np.exp(beta*np.dot(gold,theta))

def run_MCMC(n_iter, beta = 0.05, beta_incr = .02, theta_curr = [0]*num_boxes):
    # Initialize chain, maximum a posterior estimate, max gold val
    iterations = []

    for i in range(n_iter):
        # Record current iteration
        iterations.append(theta_curr)
        # Propose new theta
        theta_prop = proposal(theta_curr)
        # Set up acceptance criteria
        score_curr = eval_theta(beta, theta_curr)
        score_prop = eval_theta(beta, theta_prop)
        rho = min(1, np.divide(score_prop,score_curr))
        alpha = np.random.uniform(0,1)

        # Accept/reject theta_prop
        if rho == 1 or rho <= alpha:
            theta_curr = theta_prop
        else:
            # Propose alternate, DR sample
            theta_prop_delay = proposal(theta_prop)
            score_delay = eval_theta(beta, theta_prop_delay)
            rho_delay = min(1, np.divide(score_delay,score_curr))
            alpha_delay = np.random.uniform(0,1)
            
            # Accept/reject DR sample
            if rho_delay == 1 or rho_delay <= alpha_delay:
                theta_curr = theta_prop_delay
        
        if i%100 == 0:
          beta += beta_incr

    print(iterations)

def go(n_iter):
  run_MCMC(n_iter)

def main():
    if len(sys.argv) == 1:
        un_MCMC(sys.argv[0])
    else:
        raise Exception("Incorrect number of arguments: %s" % sys.argv)

if __name__ == '__main__':
    sys.exit(main())

