import os
import numpy as np
import torch


def count_parameters(model, trainable=False):
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def tensor2numpy(x):
    return x.cpu().data.numpy() if x.is_cuda else x.data.numpy()


def target2onehot(targets, n_classes):
    onehot = torch.zeros(targets.shape[0], n_classes).to(targets.device)
    onehot.scatter_(dim=1, index=targets.long().view(-1, 1), value=1.0)
    return onehot


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def accuracy(y_pred, y_true, nb_old, init_cls=10, increment=10):
    assert len(y_pred) == len(y_true), "Data length error."
    all_acc = {}
    all_acc["total"] = np.around(
        (y_pred == y_true).sum() * 100 / len(y_true), decimals=2
    )

    # Grouped accuracy, for initial classes
    idxes = np.where(
        np.logical_and(y_true >= 0, y_true < init_cls)
    )[0]
    label = "{}-{}".format(
        str(0).rjust(2, "0"), str(init_cls - 1).rjust(2, "0")
    )
    all_acc[label] = np.around(
        (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
    )
    # for incremental classes
    for class_id in range(init_cls, np.max(y_true), increment):
        idxes = np.where(
            np.logical_and(y_true >= class_id, y_true < class_id + increment)
        )[0]
        label = "{}-{}".format(
            str(class_id).rjust(2, "0"), str(class_id + increment - 1).rjust(2, "0")
        )
        all_acc[label] = np.around(
            (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
        )

    # Old accuracy
    idxes = np.where(y_true < nb_old)[0]

    all_acc["old"] = (
        0
        if len(idxes) == 0
        else np.around(
            (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
        )
    )

    # New accuracy
    idxes = np.where(y_true >= nb_old)[0]
    all_acc["new"] = np.around(
        (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
    )

    return all_acc


def harm_mean(seen, unseen):
    # compute from session1
    # assert len(seen) == len(unseen)
    harm_means = []
    for _seen, _unseen in zip([seen], [unseen]):
        _hmean = (2 * _seen * _unseen) / (_seen + _unseen + 1e-12)
        _hmean = float('%.3f' % (_hmean))
        harm_means.append(_hmean)
    return harm_means


def split_images_labels(imgs):
    # split trainset.imgs in ImageFolder
    images = []
    labels = []
    for item in imgs:
        images.append(item[0])
        labels.append(item[1])

    return np.array(images), np.array(labels)

def list2dict(list):
    dict = {}
    for l in list:
        s = l.split(' ')
        id = int(s[0])
        cls = s[1]
        if id not in dict.keys():
            dict[id] = cls
        else:
            raise EOFError('The same ID can only appear once')
    return dict

def text_read(file):
    with open(file, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            lines[i] = line.strip('\n')
    return lines
