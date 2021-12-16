from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch

import numpy as np
from cs285.model.knn import KNN
from torch.functional import Tensor


class QECTable(object):
    def __init__(self,
                 projection,
                 use_q_net,
                 state_dim,
                 num_actions,
                 k,
                 knn_capacity):  # KNN buffer size
        self._k = k
        # self._projection = projection
        self._num_actions = num_actions
        self.use_q_net = use_q_net

        self._knns = []  # One KNN for each action
        for i in range(num_actions):
            knn = KNN(knn_capacity, state_dim)
            self._knns.append(knn)

    def _estimate(self, state, observation, action, q_net):
        # 在缓冲区中搜索命中
        q = self._knns[action].peek(state)
        if q != None:
            # 如果命中，则返回 QEC 值。
            return q

        # 如果没有命中，用KNN求平均值（应该用Qnet）
        # print("here")
        if self.use_q_net:
          # print(q_net.predict(observation.cpu()))
          inputs = torch.unsqueeze(torch.cat((torch.squeeze(observation.cpu(), 0), torch.Tensor([action]))),0)
          # print(inputs) 
          # print(inputs.shape)
          # print(inputs)
          
          qval_qnt  = q_net(observation.cpu()).squeeze(0)[action]
          # print(qval_qnt)
          # print(f"from knn: {self._knns[action].knn_value(state.numpy(), self._k)}")
          # print(f"from q_net: {qval_qnt}")
          return qval_qnt
          # return q_net(observation.cpu()).sample().mean()
        return self._knns[action].knn_value(state.numpy(), self._k)

    def get_max_qec_action(self, observation, q_net):
        # state = self._projection.project(observation.cpu().t())
        state = observation.cpu()
        # print(f"state: {state}")
      
        # 查找并返回最大的 QEC 操作
        q = float("-inf")
        max_action = 0

        # argmax(Q(s,a))
        
        for action in range(self._num_actions):
            # 根据 QEC 表从状态和动作估计 QEC 值
            q_t = self._estimate(state, observation, action, q_net)
            # print(q_t)
            if q_t > q:
                q = q_t
                max_action = action

        # print(f"max_action: {max_action}")
        return max_action

    def update(self, observation, action, r):
        # print(observation)
        # print(observation.shape)
        # state = self._projection.project(observation.cpu().t())
        state = observation.cpu()
        # 在缓冲区中搜索命中，如果命中则更新值，如果没有命中
        # 添加条目
        action = action.int()
        # print(action.shape)
        # print(observation.shape)
        # print(f"state: {state.shape}")
        # print(f"reward: {r.shape}")
        # self._knns[action].update(observation, r)
        batch_size = action.shape[0]
        # print(f"action: {action.shape}")
        for i in range(batch_size):
          a = action[i]
          s = state[i, :]
          r_single = r[i]
          self._knns[a].update(s, r_single)






