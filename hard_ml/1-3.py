import math

import numpy as np
import torch
from torch import Tensor
from catboost.datasets import msrank_10k
from sklearn.preprocessing import StandardScaler

from typing import List


class ListNet(torch.nn.Module):
    def __init__(self, num_input_features: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.model = torch.nn.Sequential(
            torch.nn.Linear(num_input_features, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, input_1: torch.Tensor) -> torch.Tensor:
        logits = self.model(input_1)
        return logits


class Solution:
    def __init__(self, n_epochs: int = 5, listnet_hidden_dim: int = 30,
                 lr: float = 0.001, ndcg_top_k: int = 10):
        self._prepare_data()
        self.num_input_features = self.X_train.shape[1]
        self.ndcg_top_k = ndcg_top_k
        self.n_epochs = n_epochs

        self.model = self._create_model(
            self.num_input_features, listnet_hidden_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def _get_data(self) -> List[np.ndarray]:
        train_df, test_df = msrank_10k()

        X_train = train_df.drop([0, 1], axis=1).values
        y_train = train_df[0].values
        query_ids_train = train_df[1].values.astype(int)

        X_test = test_df.drop([0, 1], axis=1).values
        y_test = test_df[0].values
        query_ids_test = test_df[1].values.astype(int)

        return [X_train, y_train, query_ids_train, X_test, y_test, query_ids_test]

    def _prepare_data(self) -> None:
        (X_train, y_train, self.query_ids_train,
            X_test, y_test, self.query_ids_test) = self._get_data()
        X_train = self._scale_features_in_query_groups(X_train, self.query_ids_train)
        X_test = self._scale_features_in_query_groups(X_test, self.query_ids_test)
        self.X_train = Tensor(X_train)
        self.X_test = Tensor(X_test)
        self.ys_train = Tensor(y_train)
        self.ys_test = Tensor(y_test)

    def _scale_features_in_query_groups(self, inp_feat_array: np.ndarray,
                                        inp_query_ids: np.ndarray) -> np.ndarray:
        query_ids = list(set(inp_query_ids))
        query_ids.sort()
        scaled_list = []
        for query in query_ids:
            scaler = StandardScaler()
            scaled = scaler.fit_transform(inp_feat_array[inp_query_ids == query])
            scaled_list.append(scaled)
        return np.vstack(scaled_list)


    def _create_model(self, listnet_num_input_features: int,
                      listnet_hidden_dim: int) -> torch.nn.Module:
        torch.manual_seed(0)
        net = ListNet(listnet_num_input_features, listnet_hidden_dim)
        return net

    def fit(self) -> List[float]:
        ndcg_listg = []
        for epoch in range(self.n_epochs):
            self._train_one_epoch()
            ndcg_listg.append(self._eval_test_set())
        return ndcg_listg

    def _calc_loss(self, batch_ys: torch.FloatTensor,
                   batch_pred: torch.FloatTensor) -> torch.FloatTensor:
        return self._listnet_ce_loss(batch_ys, batch_pred)

    def _listnet_ce_loss(self, y_i, z_i):
        """
        y_i: (n_i, 1) GT
        z_i: (n_i, 1) preds
        """

        P_y_i = torch.softmax(y_i, dim=0)
        P_z_i = torch.softmax(z_i, dim=0)
        return -torch.sum(P_y_i * torch.log(P_z_i))

    def _train_one_epoch(self) -> None:
        self.model.train()
        for query in list(set(self.query_ids_train)):
            self.optimizer.zero_grad()
            data = self.X_train[self.query_ids_train == query]
            labels = torch.unsqueeze(self.ys_train[self.query_ids_train == query], 1)
            predicts = self.model(data)
            loss = self._calc_loss(labels, predicts)
            loss.backward()
            self.optimizer.step()

    def _eval_test_set(self) -> float:
        with torch.no_grad():
            self.model.eval()
            ndcgs = []
            for query in list(set(self.query_ids_test)):
                data = self.X_test[self.query_ids_test == query]
                labels = self.ys_test[self.query_ids_test == query]
                predicts = self.model(data)
                # print(labels.shape, predicts.shape)
                try:
                    ndcg = self._ndcg_k(labels, predicts.squeeze(), self.ndcg_top_k)
                except:
                    ndcg = 0.0
                if ndcg > 1 or math.isnan(ndcg) or ndcg < 0:
                    ndcg = 0.0
                ndcgs.append(ndcg)
            return np.mean(ndcgs)

    def compute_gain(self, y_value: float, gain_scheme: str = 'exp2') -> float:
        if gain_scheme == "const":
            return y_value
        elif gain_scheme == "exp2":
            return 2 ** y_value - 1


    def _dcg_k(self, ys_true: torch.Tensor, ys_pred: torch.Tensor, k: int) -> float:
        _, indices = torch.sort(ys_pred, descending=True)
        sorted_true = ys_true[indices][:k].numpy()
        gain = self.compute_gain(sorted_true)
        discount = [math.log2(float(x)) for x in range(2, len(sorted_true) + 2)]
        discounted_gain = float((gain / discount).sum())
        return discounted_gain


    def _ndcg_k(self, ys_true: torch.Tensor, ys_pred: torch.Tensor,
                ndcg_top_k: int) -> float:
        current_dcg = self._dcg_k(ys_true, ys_pred, ndcg_top_k)
        ideal_dcg = self._dcg_k(ys_true, ys_true, ndcg_top_k)
        return current_dcg / ideal_dcg