import abc
import itertools
from torch import nn
from torch.nn import functional as F
from torch import optim
from cs285.projection.random_projection import RandomProjection
from cs285.projection.vae_projection import VAEProjection
# from ... import model
import sys
sys.path.append("..")
from cs285.model.qec_table import QECTable

import numpy as np
import torch
from torch.autograd import Variable

from cs285.model.DGP import DeepGP

from cs285.infrastructure import pytorch_util as ptu
from cs285.policies.base_policy import BasePolicy
from gpytorch.mlls import DeepApproximateMLL

# import VAE
# from cs285.PyTorch_VAE.models.vanilla_vae import VanillaVAE
from gpytorch.mlls import VariationalELBO, AddedLossTerm

def create_lander_q_network(ob_dim, num_actions):
  return nn.Sequential(
        nn.Linear(ob_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, num_actions),
    )


class MLPPolicy(BasePolicy, nn.Module, metaclass=abc.ABCMeta):

    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 n_table_max,
                 use_q_net,
                 k,
                 table_capacity,
                 discrete=False,
                 learning_rate=1e-4,
                 training=True,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        # init vars
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.discrete = True
        self.size = size
        self.learning_rate = learning_rate
        self.training = training
        self.gamma = 0.99
        # self.nn_baseline = nn_baseline

        # self.vae = VanillaVAE(in_channels=self.ob_dim, latent_dim=2)
        # ob_no_simple = self.vae.encode(ob_no)

        # self.state_dim = 2
        self.state_dim = self.ob_dim

        # vae_projection = RandomProjection(self.ob_dim, self.state_dim)
        vae_projection = None
        self.q_table = QECTable(vae_projection, use_q_net, self.state_dim, self.ac_dim, k=k, knn_capacity=table_capacity)
        # print(f"in initialization: {self.q_table}")
        self.n_table_max = n_table_max

        total_dim = self.ob_dim + 1
        # self.q_net = DeepGP(total_dim)
        network_initializer = create_lander_q_network
        self.q_net = network_initializer(self.ob_dim, self.ac_dim)
        self.q_net_optimizer = torch.optim.Adam(
            [{'params': self.q_net.parameters()}]
        ,self.learning_rate)
        # self.q_net_mll = DeepApproximateMLL(VariationalELBO(self.q_net.likelihood, self.q_net, total_dim))
        
        self.loss = nn.MSELoss()
        # print(f"function {network_initializer(4)}")
        # self.q_net_target = network_initializer(self.ob_dim, self.ac_dim)



    ##################################

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    ##################################

    # query the policy with observation(s) to get selected action(s)
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        # print(f"in get action: {self.q_table}")
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None]

        observation = ptu.from_numpy(observation)
        # action_distribution = self(observation)
        # action = action_distribution.sample()  # don't bother with rsample
        action = self(observation)
        return action

    # update/train this policy
    def update(self, observations, actions, **kwargs):
        raise NotImplementedError

    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    def forward(self, observation: torch.FloatTensor):

        # observation_simple = self.vae.encode(observation)

        action = self.q_table.get_max_qec_action(observation, self.q_net)
        return action

        # if self.discrete:
        #    logits = self.logits_na(observation)
        #    action_distribution = distributions.Categorical(logits=logits)
        #    return action_distribution
        # else:
        #    batch_mean = self.mean_net(observation)
        #    scale_tril = torch.diag(torch.exp(self.logstd))
        #    batch_dim = batch_mean.shape[0]
        #    batch_scale_tril = scale_tril.repeat(batch_dim, 1, 1)
        #    action_distribution = distributions.MultivariateNormal(
        #        batch_mean,
        #        scale_tril=batch_scale_tril,
        #    )
        #    return action_distribution

#####################################################
#####################################################

class MLPPolicyMF(MLPPolicy):
    def __init__(self, ac_dim, ob_dim, n_layers, size, n_table_max, use_q_net, k, table_capacity, **kwargs):

        super().__init__(ac_dim, ob_dim, n_layers, size, n_table_max, use_q_net, k, table_capacity, **kwargs)
        self.baseline_loss = nn.MSELoss()

    def update(self, observations, actions,  next_ob_no, qvals, q_values=None):
        observations = ptu.from_numpy(observations)
        # observation_simple = self.vae.encode(observation)
        actions = ptu.from_numpy(actions)
        next_ob_no = ptu.from_numpy(next_ob_no)
        qvals = ptu.from_numpy(qvals)

        observations = observations.cpu()
        actions = actions.cpu()
        next_ob_no = next_ob_no.cpu()
        qvals = qvals.cpu()

        #TODO: update the q table
        self.q_table.update(observations, actions, qvals)

        #====================================================
        # TODO:
        #  update the q network
        #  compute the Q-values from the target network
        #---------------------------------------------------------------
        # self.q_net_optimizer.zero_grad()
        # qa_t_values = self.q_net(observations.cpu()).sample().mean()
        # # print(qa_t_values)
        # # q_t_values = torch.gather(qa_t_values, 1, actions.unsqueeze(1)).squeeze(1)
        # q_t_values = qa_t_values
        # q_t_values = Variable( q_t_values, requires_grad=True)
        # # print(next_ob_no)

        # qa_tp1_values = self.q_net_target(next_ob_no.cpu())
        # q_tp1, _ = qa_tp1_values.max(dim=1)

        # target = qvals.cpu() + self.gamma * q_tp1.cpu()
        # target = target.detach()
        # print(q_t_values)
        # print(target)
        # loss_fn = nn.MSELoss()
        # loss = loss_fn(q_t_values, target.mean())

        # loss = -self.q_net_mll(q_t_values, target)
        # self.q_net_optimizer.zero_grad()
        # loss.backward()
        # self.q_net_optimizer.step()

        #---------------------------------------------------------------
        inputs = torch.cat((observations, torch.unsqueeze(actions, 1)), 1)
        print(f"input from MLP: {inputs.shape}")
        # print(f"inputs.shape: {inputs.shape}")
        self.q_net_optimizer.zero_grad()
        # q_t_values = self.q_net(inputs.cpu())
        qa_t_values = self.q_net(observations.cpu())
        q_t_values = torch.gather(qa_t_values, 1, actions.type(torch.int64).unsqueeze(1)).squeeze(1)
        # q_t_values = torch.Tensor([qa_t_values[i, actions[i].int()] for i in range(qa_t_values.shape[0])])

        # q_t_values  = self.q_net.predict(inputs.cpu()).mean(0)
        print(f"q_t_values: {q_t_values}")
        print(f"qvals.shape: {qvals}")
        
        loss = self.loss(q_t_values, qvals)
        # print(f"q_t_values: {q_t_values.required_grad}")
        # print(f"qvals: {qvals.required_grad}")
        # loss = -self.q_net_mll(q_t_values, qvals)
        loss.backward()
        self.q_net_optimizer.step()
        #---------------------------------------------------------------

        #---------------------------------------------------------------
        # log_pi = self.q_net(observations).log_prob(actions).mean()
        # log_pi = []
        # b_size = actions.shape[0]
        # print(f"observations.shape: {observations.shape}")
        # for i in range(b_size):
        #   a = actions[i]
        #   o = observations[i, :]
        #   print(f"o: {o}")
        #   print(f"o.shape: {o.shape}")
        #   print(f"self.q_net(o.cpu()): {self.q_net(o.cpu())}")
        #  log_p_i = self.q_net(o.cpu()).log_prob(a.cpu())
        #  log_pi.append(log_p_i.meam())
        
        # loss = -(log_pi * qvals_std).sum()
        # print(f"old loss: {-(log_pi * qvals_std).sum()}")
        # return q_net_optimizer.item()
        #====================================================
        train_log = {
            'Training Loss': ptu.to_numpy(loss),
        }
        return train_log

    def update_target_network(self):
        for target_param, param in zip(
                self.q_net_target.parameters(), self.q_net.parameters()
        ):
            target_param.data.copy_(param.data)

