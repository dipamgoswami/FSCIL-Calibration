import copy
import logging
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from utils.toolkit import tensor2numpy, accuracy, harm_mean
from scipy.spatial.distance import cdist
import torch.nn.functional as F

EPSILON = 1e-8
batch_size = 64

class BaseLearner(object):
    def __init__(self, args):
        self._cur_task = -1
        self._known_classes = 0
        self._total_classes = 0
        self._network = None
        self._old_network = None
        self._data_memory, self._targets_memory = np.array([]), np.array([])
        self.topk = 5

        self._memory_size = args["memory_size"]
        self._device = args["device"][0]
        self._multiple_gpus = args["device"]
        self.args = args

    @property
    def feature_dim(self):
        if isinstance(self._network, nn.DataParallel):
            return self._network.module.feature_dim
        else:
            return self._network.feature_dim
    
    def save_checkpoint(self, filename):
        self._network.cpu()
        save_dict = {
            "tasks": self._cur_task,
            "model_state_dict": self._network.state_dict(),
        }
        torch.save(save_dict, "{}_{}.pkl".format(filename, self._cur_task))

    def after_task(self):
        pass

    def _evaluate(self, y_pred, y_true):
        ret = {}
        grouped = accuracy(y_pred.T[0], y_true, self._known_classes, self.args["init_cls"], self.args["increment"])
        ret["hm"] = harm_mean(grouped["old"],grouped["new"])
        ret["grouped"] = grouped
        ret["top1"] = grouped["total"]
        ret["top{}".format(self.topk)] = np.around(
            (y_pred.T == np.tile(y_true, (self.topk, 1))).sum() * 100 / len(y_true),
            decimals=2,
        )
        return ret

    def eval_task(self):
        if self.args["fecam"]:
            y_pred, y_true = self._eval_maha(self.test_loader, self._network.fc.weight.data, self.cov_mats)
        else:
            y_pred, y_true = self._eval_cnn(self.test_loader)
        
        cnn_accy = self._evaluate(y_pred, y_true)
        nme_accy = None

        return cnn_accy, nme_accy

    def incremental_train(self):
        pass

    def _train(self):
        pass
    
    def _get_memory(self):
        if len(self._data_memory) == 0:
            return None
        else:
            return (self._data_memory, self._targets_memory)

    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs)["logits"]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = self._network(inputs)["logits"]
            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True
            )[
                1
            ]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]
    
    def _eval_maha(self, loader, class_means, cov_mats):
        self._network.eval()
        vectors, y_true = self._extract_vectors(loader)
        dists = self._maha_dist(vectors, class_means, cov_mats)
        scores = dists.T  # [N, nb_classes], choose the one with the smallest distance
        return np.argsort(scores, axis=1)[:, :1], y_true  # [N, topk]

    def _maha_dist(self, vectors, class_means, cov_mats):
        vectors = torch.tensor(vectors).to(self._device)
        maha_dist = []
        for class_index in range(len(class_means)):
            dist = self._mahalanobis(vectors, class_means[class_index], cov_mats[class_index])     
            maha_dist.append(dist)
        maha_dist = np.array(maha_dist)  # [nb_classes, N]  
        return maha_dist

    def _mahalanobis(self, vectors, class_means, cov=None):
        x_minus_mu = F.normalize(vectors, p=2, dim=-1) - F.normalize(class_means, p=2, dim=-1)
        if cov is None:
            cov = torch.eye(self._network.feature_dim)  # identity covariance matrix for euclidean distance
        inv_covmat = torch.linalg.pinv(cov).float().to(self._device)
        left_term = torch.matmul(x_minus_mu, inv_covmat)
        mahal = torch.matmul(left_term, x_minus_mu.T)
        return torch.diagonal(mahal, 0).cpu().numpy()
    
    def shrink_cov(self, cov, alpha):
        iden = torch.eye(cov.shape[0])
        cov_ = cov + (alpha*iden)
        return cov_


    def _extract_vectors(self, loader):
        self._network.eval()
        vectors, targets = [], []

        with torch.no_grad():
            for _, _inputs, _targets in loader:
                _targets = _targets.numpy()
                if isinstance(self._network, nn.DataParallel):
                    _vectors = tensor2numpy(
                        self._network.module.extract_vector(_inputs.to(self._device))
                    )
                else:
                    _vectors = tensor2numpy(
                        self._network.extract_vector(_inputs.to(self._device))
                    )

                vectors.append(_vectors)
                targets.append(_targets)

        return np.concatenate(vectors), np.concatenate(targets)
    