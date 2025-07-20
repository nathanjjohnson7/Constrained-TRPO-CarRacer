#THIS WAS UNSUCCESSFUL
#I subsequently implemented CRPO (crpo_trpo.py) and it worked quite well

import gymnasium as gym
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F

import numpy as np

import imageio

log_file = open("game_scores.txt", "a")

MAX_EPISODE_STEPS = 1000

#inspired by https://pages.stat.wisc.edu/~wahba/stat860public/pdf1/cj.pdf
# https://optimization.cbe.cornell.edu/index.php?title=Conjugate_gradient_methods
#solve for x: Ax = b
def conjugate_gradient(A_fn, b, nsteps=10, residual_tol=1e-10):
    x = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()
    rs_old = torch.dot(r,r)
    for i in range(nsteps):
        Ap = A_fn(p)
        pAp = torch.dot(p, Ap)
        alpha = rs_old/pAp
        x += alpha * p
        r -= alpha * Ap
        rs_new = torch.dot(r,r)

        if rs_new.sqrt().item() < residual_tol:
            break

        p=r+rs_new/rs_old*p
        rs_old=rs_new

    return x

class Actor(nn.Module):
    def __init__(self, input_dims=10, hidden=128, n_actions=5):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(input_dims, hidden)
        self.linear2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, n_actions)
        self.out.weight.data.mul_(0.1)
        self.out.bias.data.mul_(0.0)
    
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.out(x)
        return x

class Critic(nn.Module):
    def __init__(self, input_dims=10, hidden=128, output_dims=1):
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(input_dims, hidden)
        self.linear2 = nn.Linear(hidden, hidden)
        self.reward_critic = nn.Linear(hidden, output_dims)
        self.constraint_critic = nn.Linear(hidden, output_dims)
    
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        reward_value = self.reward_critic(x)
        constraint_value = self.constraint_critic(x)
        return reward_value, constraint_value

#Welford's Online Algorithm: 
#adapted from: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
#we use this to track the running count, mean and squared distance from the mean, for observation values
def update_aggregate_stats(aggregate, new_value):
    #aggregate.shape-> [3, N]
    #new_value.shape-> [N]
    #N denotes observation size

    count, mean, M2 = aggregate[0], aggregate[1], aggregate[2]

    count += 1
    delta = new_value - mean
    mean += delta / count
    delta2 = new_value - mean
    M2 += delta * delta2

    new_aggregate = torch.stack([count, mean, M2], dim=0)

    return new_aggregate

#Welford's Online Algorithm: 
#adapted from: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
# Retrieve the mean, variance, sample variance, std and stample_std from an aggregate
def finalize_aggregate_stats(aggregate):
    count, mean, M2 = aggregate[0], aggregate[1], aggregate[2]
    if count[0] == 1: #can't have an std with just one value
        return mean, None, None, None, None
    variance = M2 / count
    sample_variance = M2 / (count - 1)
    std = torch.sqrt(variance)
    sample_std = torch.sqrt(sample_variance)
    return mean, variance, sample_variance, std, sample_std


class Agent:
    def __init__(self, nsteps=1024, gamma=0.99, lam=0.95, lr=0.001):
        self.nsteps = nsteps

        self.rewards = []
        self.reward_values = []
        self.constraint_costs = []
        self.constraint_values = []
        self.dones = []
        self.log_probs = []
        self.policies = []
        self.states = []
        self.actions = []

        self.reward_adv = None
        self.cc_adv = None

        self.gamma = gamma
        self.lam = lam

        #used to track running mean and std of features
        self.aggregate = None

        self.actor = Actor()
        self.critic = Critic()
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr = lr)

        self.constraint_limit = 1
        self.tolerance = 0.5

        #max kl divergence per update
        self.delta = 0.1

        self.env = gym.make("CarRacing-v3", continuous=False, max_episode_steps=MAX_EPISODE_STEPS)#, render_mode="human")
        self.reset_env()

        #track rewards over the whole game, across all rollouts of a game. 
        self.running_reward_counter = 0
        self.rewards_per_game = []
        self.feasible_game_rewards = []

        self.n_critic_updates = 15

        self.current_images = []

        self.action_taken_count = torch.zeros((5))

        self.entropy_coeff = 0.01

    def reset_env(self):
        #due to computational constraints, we just train on one track.
        obs, info = self.env.reset(seed=10)

        # 0: do nothing
        # 1: steer right
        # 2: steer left
        # 3: gas
        # 4: brake

        #move closer to the first turn so we get an idea if the model will work, quicker
        for i in range(70):
            obs, reward, done, truncated, info = self.env.step(int(3))
        for i in range(20): 
            obs, reward, done, truncated, info = self.env.step(int(4))

        print("ready")

        #first value is off_track_flag, this value can easily be inferred from position data, so we exclude it
        self.frames = [info["data"][1:]]

        if(self.aggregate is None):
            data = torch.tensor(info["data"][1:], dtype=torch.float32)
            #stack count=1, mean, and squared distance from mean for first obs
            self.aggregate = torch.stack([torch.ones_like(data), data, torch.zeros_like(data)], dim=0)

    def rollout(self):
        #reset everything
        self.rewards = []
        self.reward_values = []
        self.constraint_costs = []
        self.constraint_values = []
        self.dones = []
        self.log_probs = []
        self.policies = []
        self.states = []
        self.actions = []

        done = False
        while not done:
            #we only use one frame
            state = torch.tensor(self.frames[0], dtype=torch.float32)

            #Create a mask to skip indices 4 and 5 - the sine and cosine of the car angle relative to the trajectory of the track
            #We don't want to normalize these separately with a running mean and std since sine and cosine values should add to 1
            mask = torch.ones_like(state, dtype=torch.bool)
            mask[4] = False
            mask[5] = False

            #normalize other values using running mean and std of each value
            obs_mean, _, _, obs_std, _ = finalize_aggregate_stats(self.aggregate)
            if obs_std is not None:
                obs_std = obs_std.clamp(min=1e-8)
                state[mask] = (state[mask] - obs_mean[mask]) / obs_std[mask]

            else:
                state[mask] = state[mask] - obs_mean[mask]
            state = state.unsqueeze(0)

            policy = self.actor(state)
            dist = torch.distributions.Categorical(logits=policy)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            self.action_taken_count[action.item()] += 1
            
            obs, reward, terminated, truncated, info = self.env.step(action.item())

            self.current_images.append(obs)

            done = terminated or truncated

            #we add a penalty for cars that drive off the map, ending the episode early
            #we give a 0.1 penalty for the remaining timesteps
            #90 is subtracted since we used 90 steps to position the car closer to the first turn.
            if terminated and info["lap_finished"] == False:
                reward += -0.1 * (MAX_EPISODE_STEPS - 90 - len(self.current_images))

            self.frames = [info["data"][1:]]

            self.rewards.append(reward)
            self.constraint_costs.append(info["data"][0])
            self.dones.append(done)
            self.log_probs.append(log_prob)
            self.policies.append(policy)
            self.states.append(state)
            self.actions.append(action)           

            self.running_reward_counter += reward

            if done:
                self.rewards_per_game.append(self.running_reward_counter)
                if sum(self.constraint_costs) == 0:
                    self.feasible_game_rewards.append(self.running_reward_counter)
                print(f"Game {len(self.rewards_per_game)}: {self.running_reward_counter}")
                log_file.write(f"{len(self.rewards_per_game)}, {self.running_reward_counter}, {sum(self.constraint_costs)}\n")
                log_file.flush()
                #we only want to save the model if it adheres to constraints
                if self.running_reward_counter == max(self.feasible_game_rewards):
                    torch.save(self.actor.state_dict(), 'best_actor.pt')
                    torch.save(self.critic.state_dict(), 'best_critic.pt')
                    torch.save(self.aggregate, "aggregate_stats.pt")
                    imageio.mimsave(f"videos\car_racer4_video_{len(self.rewards_per_game)}_{int(sum(self.rewards))}_best.mp4", self.current_images, fps=30)
                else:
                    imageio.mimsave(f"videos\car_racer4_video_last.mp4", self.current_images, fps=30)
                self.running_reward_counter = 0
                self.reset_env()
                self.current_images = []

                print(self.action_taken_count)
                self.action_taken_count = torch.zeros((5))
                break #each batch contains only one game
            else:
                self.aggregate = update_aggregate_stats(self.aggregate, torch.tensor(info["data"][1:], dtype=torch.float32))

        #Convert lists to tensors
        self.rewards = torch.tensor(self.rewards, dtype=torch.float32)
        self.constraint_costs = torch.tensor(self.constraint_costs, dtype=torch.float32)
        self.dones = torch.tensor(self.dones, dtype=torch.float32)
        self.log_probs = torch.cat(self.log_probs)
        self.policies = torch.cat(self.policies)
        self.states = torch.cat(self.states)
        self.actions = torch.cat(self.actions)
        
        #nsteps denotes the number of states. 
        # the above arrays hold states and next states and the last one is just a next state
        self.nsteps = self.rewards.shape[0]-1 

    #create a model from flattened parameters
    @staticmethod
    def model_from_flattened(model, params_vec):
        i = 0
        for p in model.parameters():
            length = p.numel()
            p.data.copy_(params_vec[i:i + length].view_as(p))
            i += length

    def get_kl_divergence(self, fraction=0.1):
        states_subset = self.states[:-1] #last one is next state that isn't used
        n_total = states_subset.shape[0]
        #we only use 10% of the states to estimate the kl divergence, or 100 states if the batch
        # has less than 1000, or all the states, if there are less than 100 states
        #this needs to be done since every batch consists of exactlly one episode in this implementation
        num_samples = min(max(int(n_total * fraction), 100), n_total)
        if fraction < 1.0:
            indices = torch.randperm(states_subset.shape[0])[:num_samples]
            states_subset = states_subset[indices]

        #Note: the parameters of actor haven't changed since the last rollout
        #The computation graph gets consumed after gradient calls so we make a forward pass each time kl divergence is called
        #Here we are computing the kl divergence of the policy with itself, since this is for a taylor approximation
        new_policy = self.actor(states_subset)
        old_policy = new_policy.detach()

        new_probs = F.softmax(new_policy, dim=-1)
        old_probs = F.softmax(old_policy, dim=-1)

        eps = 1e-8
        kl = new_probs * (torch.log(new_probs+eps) - torch.log(old_probs+eps))

        kl = kl.sum(dim=1) #sum over actions: shape-> [n_steps]
        return kl.mean() #average over timesteps
    
    #inspired by: https://github.com/ikostrikov/pytorch-trpo/blob/master/trpo.py  
    #this function uses pearlmutters's trick to avoid calculating the complete hessian of the kl divergence
    #instead it calculates the hessian of the product of the kl divergence and a vector p  
    def A_fn(self, p):
        kl = self.get_kl_divergence()

        #gradient of kl divergence
        grad_kl = torch.autograd.grad(kl, self.actor.parameters(), create_graph=True, allow_unused=True)
        grad_kl = [g if g is not None else torch.zeros_like(p) for g, p in zip(grad_kl, self.actor.parameters())]
        grad_kl = torch.cat([g.reshape(-1) for g in grad_kl])

        #directional derivative
        grad_klp = (grad_kl * p.detach()).sum()

        #Hessian-vector product
        hessian_klp = torch.autograd.grad(grad_klp, self.actor.parameters(), allow_unused=True)
        hessian_klp = [g if g is not None else torch.zeros_like(p) for g, p in zip(hessian_klp, self.actor.parameters())]
        hessian_klp = torch.cat([g.contiguous().view(-1) for g in hessian_klp])

        damping = 0.1 #from spinning up RL TRPO page
        return hessian_klp + damping * p # Add damping for numerical stability

    def discounted_returns(self, returns):
        disc_return = torch.zeros((self.nsteps+1))
        #loop backward over nsteps
        for i in range(self.nsteps-1, -1, -1):
            #when the next state is a terminal state, discount becomes zero
            discount = self.gamma*(1-self.dones[i+1])
            disc_return[i] = returns[i] + discount*disc_return[i+1]
        return disc_return[:-1].detach()

    #get the generalized advantage estimation (gae)
    def calculate_gae(self):
        self.reward_values, self.constraint_values = self.critic(self.states)

        #reward deltas
        deltas_r = torch.zeros((self.nsteps))

        #constraint cost deltas
        deltas_cc = torch.zeros((self.nsteps))

        for i in range(self.nsteps):
            deltas_r[i] = (self.rewards[i] + 
                           self.gamma*self.reward_values[i+1] -
                           self.reward_values[i])
            
            deltas_cc[i] = (self.constraint_costs[i] + 
                            self.gamma*self.constraint_values[i+1] -
                            self.constraint_values[i])

        gae_reward = torch.zeros((self.nsteps+1))
        gae_cc = torch.zeros((self.nsteps+1))

        #loop backward over nsteps
        for i in range(self.nsteps-1, -1, -1):
            #when the next state is a terminal state, discount becomes zero
            discount_r = self.gamma*self.lam_r*(1-self.dones[i+1])
            discount_cc = self.gamma*self.lam_cc*(1-self.dones[i+1])

            gae_reward[i] = deltas_r[i] + discount_r*gae_reward[i+1]

            gae_cc[i] = deltas_cc[i] + discount_cc*gae_cc[i+1]

        return gae_reward[:-1].detach(), gae_cc[:-1].detach()

    #gradient descent for critic
    def update_critics(self):
        self.reward_values, self.constraint_values = self.critic(self.states)

        discounted_rewards = self.discounted_returns(self.rewards)
        reward_critic_loss = F.mse_loss(self.reward_values[:-1].squeeze(-1), discounted_rewards)

        discounted_constraint_costs = self.discounted_returns(self.constraint_costs)
        constraint_critic_loss = F.mse_loss(self.constraint_values[:-1].squeeze(-1), discounted_constraint_costs)

        loss = reward_critic_loss + constraint_critic_loss

        print("critic loss: ", loss)
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step() 
    
    def get_vars_for_langrangian(self):
        reward_adv, cc_adv = self.calculate_gae()

        #normalize advantages
        reward_adv = (reward_adv - reward_adv.mean()) / (reward_adv.std() + 1e-8)
        cc_adv = (cc_adv - cc_adv.mean()) / (cc_adv.std() + 1e-8)

        self.reward_adv = reward_adv
        self.cc_adv = cc_adv

        dist = torch.distributions.Categorical(logits=self.policies)
        entropy = dist.entropy()[:-1]
        print("entropy  ", entropy.mean())

        self.actor.zero_grad()

        x = (reward_adv.detach() * self.log_probs[:-1]).mean() + self.entropy_coeff * entropy.mean()
        x.backward(retain_graph = True) #retain graph so we can use the same gradients to calculate b
        grads_reward = torch.cat([
            p.grad.clone().view(-1) if p.grad is not None 
            else torch.zeros_like(p).view(-1) 
            for p in self.actor.parameters()
        ])
        g = grads_reward #gradient of objective

        self.actor.zero_grad()

        x = (cc_adv.detach() * self.log_probs[:-1]).mean()
        x.backward()
        grads_cc = torch.cat([
            p.grad.clone().view(-1) if p.grad is not None 
            else torch.zeros_like(p).view(-1) 
            for p in self.actor.parameters()
        ])
        b = grads_cc #gradient of constraint

        constraint_returns = self.discounted_returns(self.constraint_costs)
        c = constraint_returns.mean() - self.constraint_limit

        #inverse of the hessian of the KL divergence * gradient of the reward advantage
        h_inverse_g = conjugate_gradient(self.A_fn, g)

        #inverse of the hessian of the KL divergence * gradient of the constraint cost advantage
        h_inverse_b = conjugate_gradient(self.A_fn, b)

        return g, c, b, h_inverse_g, h_inverse_b

    def line_search(self, params, step, g, c, b, max_backtracks=100, alpha=0.8, infeasible=False):
        new_model = Actor()

        frac = 1.0
        for i in range(max_backtracks):
            diff = frac * step
            params_new = params + diff
            #return params_new

            trust_region_constraint = 0.5 * diff.dot(self.A_fn(diff))
            print("trust_region_constraint:", trust_region_constraint)

            if trust_region_constraint > self.delta:
                print("trust region constraint unmet. Backtracing ...")
                frac *= alpha
                continue

            cost_constraint = c + b.dot(diff)
            print("cost_constraint_old: ",c, " cost_constraint_new:", cost_constraint)

            if cost_constraint > self.tolerance:
                print("cost constraint unmet. Backtracing ...")
                frac *= alpha
                continue

            #Performance check (only if constraints are satisfied)
            self.model_from_flattened(new_model, params_new)
            policy = new_model(self.states)
            dist = torch.distributions.Categorical(logits=policy)
            new_log_prob = dist.log_prob(self.actions)

            ratio = torch.exp(new_log_prob[:-1] - self.log_probs[:-1])
            real_performance = (self.reward_adv * ratio).mean()
            estimated_performance = g.dot(diff)

            print("Real performance:", real_performance)
            print("Estimated performance:", estimated_performance)

            if estimated_performance.abs() > 1e-8 and real_performance / estimated_performance < 0.1:
                print("Performance check failed. Backtracking ...")
                frac *= alpha
                continue

            #All checks passed
            print("Step accepted at backtrack:", i, " diff sum:", diff.sum())
            return params_new

        #No feasible step found
        print("--------- Couldn't find feasible model during backtracking line_search")
        return params+step

    def solve_dual(self):
        print("Solving ...")
        infeasible = False

        g, c, b, h_inverse_g, h_inverse_b = self.get_vars_for_langrangian()

        q = g.dot(h_inverse_g)
        r = g.dot(h_inverse_b)
        s = b.dot(h_inverse_b)

        eps = 1e-8

        if (c**2)/s - self.delta>0 and c>0:
            print("--------------------- Infeasible optimization")
            infeasible = True

            #solution to \min_\theta c+b(\theta - \theta_k) s.t. \frac{1}{2} (\theta - \theta_k)H(\theta - \theta_k) \le \delta
            #min theta: c+b(theta-theta_k) s.t. (theta-theta_k)H(theta-theta_k) <= delta
            #solving using the lagrangian formulation gives us
            search_direction = -torch.sqrt(2*self.delta/(s+eps))*h_inverse_b
            #see Appendix C of TRPO Paper
            max_step_length = torch.sqrt(2*self.delta/search_direction.dot(self.A_fn(search_direction)))

            step = search_direction# * max_step_length

        else:
            lambda_a = torch.sqrt((q-(r**2)/(s+eps))/(self.delta-(c**2)/(s+eps)))
            lambda_b = torch.sqrt(q/self.delta) #if nu is zero

            if c > 0:
                lambda_a_proj = torch.max(lambda_a, torch.max((r/(c+eps))+eps, torch.tensor(0.0)))
                lambda_b_proj = torch.min(r/(c+eps), torch.max(lambda_b, torch.tensor(0.0)))
            else:
                lambda_a_proj = torch.min((r/(c+eps))-eps, torch.max(lambda_a, torch.tensor(0.0)))
                lambda_b_proj = torch.max(lambda_b, torch.max(r/(c+eps), torch.tensor(0.0)))

            f_a_lambda = lambda lam: (1/(2*lam))*((r**2)/s - q) + (lam/2)*((c**2)/s - self.delta) - (r*c)/s
            f_b_lambda = lambda lam: (-1/2)*((q/lam) + (lam*self.delta))

            lambda_opt = lambda_a_proj if f_a_lambda(lambda_a_proj) >= f_b_lambda(lambda_b_proj) else lambda_b_proj

            nu_opt = torch.max((lambda_opt*c - r) / (s+eps), torch.tensor(0.0))
            
            search_direction = (h_inverse_g - h_inverse_b*nu_opt)/(lambda_opt+eps)

            kl_constraint_max_step = torch.sqrt(2*self.delta/search_direction.dot(self.A_fn(search_direction)))

            #cost constraint-> c + b(theta-theta_k) <= 0
            #let theta-theta_k = beta*search_direction   (beta is the max step size)
            #c + b(beta*search_direction) <= 0
            #beta <= -c/(b*search_direction)
            cost_constraint_max_step = -c/(b.dot(search_direction))

            #we want the lowest max step size of the two constraints, so we don't overstep any constraint
            max_step_length = torch.min(kl_constraint_max_step, cost_constraint_max_step)

            step = search_direction #* max_step_length


        print("search direction: ", search_direction, search_direction.sum())

        #flatten model parameters
        params = list(self.actor.parameters())
        flat_params = torch.cat([p.detach().view(-1) for p in params])

        #line search
        new_params = self.line_search(flat_params, step, g, c, b, infeasible=infeasible)

        self.model_from_flattened(self.actor, new_params)

    def update(self):
        self.rollout()
        self.solve_dual()
        for _ in range(self.n_critic_updates):
            self.update_critics()

if __name__ == "__main__":
    agent = Agent()
    for i in range(10000000):
        print(f"\nUpdate: {i}\n")
        agent.update()
