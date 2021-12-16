import numpy as np

from .base_agent import BaseAgent
from cs285.policies.MLP_policy import MLPPolicyMF
from cs285.infrastructure.replay_buffer import ReplayBuffer

class MFAgent(BaseAgent):
    def __init__(self, env, agent_params):
        super(MFAgent, self).__init__()
        # print(agent_params)

        # init vars
        self.env = env
        self.agent_params = agent_params
        self.gamma = self.agent_params['gamma']
        self.standardize_advantages = self.agent_params['standardize_advantages']
        # self.nn_baseline = self.agent_params['nn_baseline']
        self.reward_to_go = self.agent_params['reward_to_go']
        # self.gae_lambda = self.agent_params['gae_lambda']

        # actor/policy
        self.actor = MLPPolicyMF(
            self.agent_params['ac_dim'],
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            self.agent_params['n_table_max'],
            self.agent_params['use_q_net'],
            self.agent_params['k'],
            self.agent_params['table_capacity'],
            # self.agent_params['q_func'],
            discrete=self.agent_params['discrete'],
            learning_rate=self.agent_params['learning_rate']
            # nn_baseline=self.agent_params['nn_baseline'],
        )

        # replay buffer
        self.replay_buffer = ReplayBuffer(1000000)

    def train(self, observations, actions, rewards_list, next_observations, terminals):

        """
            Training a MF agent refers to updating its actor using the given observations/actions
            and the calculated qvals/advantages that come from the seen rewards.
        """

        ## TODO: VAE encoder to convert continous states (observations) to discrete
        ## TODO:

        # update the PG actor/policy using the given batch of data, and
        # return the train_log obtained from updating the policy

        # HINT1: use helper functions to compute qvals and advantages
        # HINT2: look at the MLPPolicyPG class for how to update the policy
            # and obtain a train_log
        qvals = self.calculate_q_vals(rewards_list)
        std_qvals = self.standardize_qvals(observations, rewards_list, qvals, terminals)
        train_log = self.actor.update(observations, actions,  next_observations, std_qvals, q_values=qvals)

        return train_log

    def calculate_q_vals(self, rewards_list):

        """
            Monte Carlo estimation of the Q function.
        """
        if not self.reward_to_go:
            discounted_rewards = [self._discounted_return(reward) for reward in rewards_list]
            q_values = np.concatenate(discounted_rewards).ravel()
            print(type(q_values))

        # Estimate Q^{pi}(s_t, a_t) by the discounted sum of rewards starting from t
        else:
            discounted_cusum = [self._discounted_cumsum(reward) for reward in rewards_list]
            q_values = np.concatenate(discounted_cusum).ravel()

        return q_values

        
    def standardize_qvals(self, obs, rews_list, q_values, terminals):

        """
            Computes advantages by (possibly) using GAE, or subtracting a baseline from the estimated Q values
        """

        # Estimate the advantage when nn_baseline is True,
        # by querying the neural network that you're using to learn the value function

        advantages = q_values.copy()

        # Normalize the resulting advantages
        if self.standardize_advantages:
            ## TODO: standardize the advantages to have a mean of zero
            ## and a standard deviation of one
            advantages = (advantages - np.mean(advantages))/np.std(advantages)

        return advantages

    #####################################################
    #####################################################

    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)

    def sample(self, batch_size):
        return self.replay_buffer.sample_recent_data(batch_size, concat_rew=False)

    #####################################################
    ################## HELPER FUNCTIONS #################
    #####################################################

    def _discounted_return(self, rewards):
        """
            Helper function

            Input: list of rewards {r_0, r_1, ..., r_t', ... r_T} from a single rollout of length T

            Output: list where each index t contains sum_{t'=0}^T gamma^t' r_{t'}
        """

        # TODO: create list_of_discounted_returns
        T = len(rewards)
        gammas = np.array([self.gamma**i for i in range(T)])
        v = sum(gammas * rewards)
        list_of_discounted_returns = np.array([v]*T)

        return list_of_discounted_returns

    def _discounted_cumsum(self, rewards):
        """
            Helper function which
            -takes a list of rewards {r_0, r_1, ..., r_t', ... r_T},
            -and returns a list where the entry in each index t' is sum_{t'=t}^T gamma^(t'-t) * r_{t'}
        """
        # TODO: create `list_of_discounted_returns`
        # HINT: it is possible to write a vectorized solution, but a solution
            # using a for loop is also fine
        T = len(rewards)

        gammas = np.array([[0 for j in range(t)] + [self.gamma**i for i in range(T-t)] for t in range(T)])
        m = np.dot(gammas, rewards)

        list_of_discounted_cumsums = m

        return list_of_discounted_cumsums
