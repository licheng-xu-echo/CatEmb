import torch
from torch import nn
import torch.nn.functional as F
def cycle_index(num, shift):
    arr = torch.arange(num) + shift
    arr[-shift:] = torch.arange(shift)
    return arr
def do_CL(X, Y, normalize, metric, T, lambda_cl, lambda_kl, CL_neg_samples):
    """
    :param X: "input"
    :param Y: "target"
    :param normalize: bool, l2 or not
    :param metric: InfoNCE
    :param T: temperature
    :param lambda_cl: loss, weight
    :param lambda_kl: loss, weight
    :param CL_neg_samples:
    :return:
    """
    # default, l2 normalization
    if normalize:
        X = F.normalize(X, dim=-1)
        Y = F.normalize(Y, dim=-1)
    CL_loss, KL_loss = torch.tensor(0.), torch.tensor(0.)
    if lambda_cl > 0.:
        if metric == 'InfoNCE_dot_prod':
            criterion = nn.CrossEntropyLoss()
            B = X.size()[0]
            logits = torch.mm(X, Y.transpose(1, 0))  # B*B
            logits = torch.div(logits, T)
            labels = torch.arange(B, dtype=torch.long).to(logits.device)  # B*1

            CL_loss = criterion(logits, labels)
            # pred = logits.argmax(dim=1, keepdim=False)
            # CL_acc = pred.eq(labels).sum().detach().cpu().item() * 1. / B

        elif metric == 'EBM_dot_prod':
            criterion = nn.BCEWithLogitsLoss()
            neg_Y = torch.cat([Y[cycle_index(len(Y), i + 1)]
                               for i in range(CL_neg_samples)], dim=0)
            neg_X = X.repeat((CL_neg_samples, 1))

            pred_pos = torch.sum(X * Y, dim=1) / T
            pred_neg = torch.sum(neg_X * neg_Y, dim=1) / T

            loss_pos = criterion(pred_pos, torch.ones(len(pred_pos)).to(pred_pos.device))
            loss_neg = criterion(pred_neg, torch.zeros(len(pred_neg)).to(pred_neg.device))
            CL_loss = loss_pos + CL_neg_samples * loss_neg

            # CL_acc = (torch.sum(pred_pos > 0).float() +
            #           torch.sum(pred_neg < 0).float()) / \
            #          (len(pred_pos) + len(pred_neg))
            # CL_acc = CL_acc.detach().cpu().item()

        else:
            raise Exception

    # apply KL-divergence
    if lambda_kl > 0.:
        kl_criterion = nn.KLDivLoss(reduction="batchmean", log_target=True)
        KL_loss = kl_criterion(F.log_softmax(X, dim=-1), F.log_softmax(Y, dim=-1))

    return CL_loss, KL_loss


def dual_CL(X, Y, normalize=True, metric="InfoNCE_dot_prod", T=0.1, lambda_cl=1., lambda_kl=-1., CL_neg_samples=1):
    # two sides
    CL_loss_1, KL_loss_1 = do_CL(X, Y, normalize, metric, T, lambda_cl, lambda_kl, CL_neg_samples)
    CL_loss_2, KL_loss_2 = do_CL(Y, X, normalize, metric, T, lambda_cl, lambda_kl, CL_neg_samples)
    # return (CL_loss_1 + CL_loss_2) / 2, (CL_acc_1 + CL_acc_2) / 2
    return (CL_loss_1 + CL_loss_2) / 2, (KL_loss_1 + KL_loss_2) / 2