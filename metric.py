import torch


def accuracy(y_pred, y):
    """computes classification accuracy

    Parameters
    ----------
    y_pred: torch.tensor with size BZxK

    y: 1-D torch.torch.tensor with size BZ

    Returns
    -------
        * float

    """
    _, predicted = torch.max(y_pred, 1)
    correct = (predicted == y).float()
    acc = correct.sum() / len(y)
    return acc
