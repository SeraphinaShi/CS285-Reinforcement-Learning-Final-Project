import numpy as np
from sklearn.neighbors import KDTree


class KNN:
    """ K Nearest-Neighbor with LRU. """
    def __init__(self, capacity, state_dim):
        self._capacity = capacity
        self._states = np.zeros((capacity, state_dim))
        self._q_values = np.zeros(capacity)
        self._lru = np.zeros(capacity)
        self._current_capacity = 0
        self._time = 0.0
        self._tree = None

    def peek(self, state):
        index = self._peek_index(state)
        if index >= 0:
            return self._q_values[index]
        return None

    def update(self, state, r):
        # print(state)
        index = self._peek_index(state)
        if index >= 0:
            # 如果准确命中
            max_q = max(self._q_values[index], r)
            self._q_values[index] = max_q
        else:
            # 如果没有找到，添加一个条目
            self._add(state, r)

    def knn_value(self, state, k):
        # print(f"self._current_capacity: {self._current_capacity}")
        # print(f"k: {k}")
        if self._current_capacity < k:
            return 0.0
        # print(f"knn_value(self, state, k), state: {state}")
        # print(f"knn_value(self, state, k), k: {k}")
        # print(f"self._tree.query([state], k=k), [state]: {[state]}")
        #_, indices = self._tree.query([state], k=k)
        _, indices = self._tree.query(state, k=k)

        size = len(indices[0])
        if size == 0:
            return 0.0

        value = 0.0
        for index in indices[0]:
            value += self._q_values[index]
            self._lru[index] = self._time
            self._time += 0.01

        return value / size

    def _peek_index(self, state):
        if self._current_capacity == 0:
            return -1

        # 我会寻找最近的 
        state = np.squeeze(state)
        # print(f"state_after_squeeze: {[state.numpy()]}")
        _, indices = self._tree.query([state.numpy()], k=1)
        index = indices[0][0]

        if np.allclose(self._states[index], state):
            # 最接近的一个是一样的
            self._lru[index] = self._time
            self._time += 0.01
            return index
        else:
            return -1

    def _add(self, state, r):
        # print(f"self._current_capacity: {self._current_capacity}")
        # print(f"self._capacity: {self._capacity}")
        if self._current_capacity >= self._capacity:
            # find the LRU entry
            old_index = np.argmin(self._lru)# 线性搜索
            self._states[old_index] = state
            self._q_values[old_index] = r
            self._lru[old_index] = self._time
        else:
            # print(self._states[self._current_capacity].shape)
            # print(f"state: {state.shape}")
            self._states[self._current_capacity] = state
            self._q_values[self._current_capacity] = r
            self._lru[self._current_capacity] = self._time
            self._current_capacity += 1

        self._time += 0.01
        # 重建树
        # print(f"tree: {self._states[:self._current_capacity]}")
        self._tree = KDTree(self._states[:self._current_capacity])



