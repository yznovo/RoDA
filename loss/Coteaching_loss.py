import torch
import torch.nn.functional as F
import numpy as np

# def loss_coteaching(y_1, y_2, t, forget_rate):
#     loss_1 = F.cross_entropy(y_1, t, reduce = False)
#     ind_1_sorted = np.argsort(loss_1.data.cpu())
#     loss_1_sorted = loss_1[ind_1_sorted]
#
#     loss_2 = F.cross_entropy(y_2, t, reduce = False)
#     ind_2_sorted = np.argsort(loss_2.data.cpu())
#     loss_2_sorted = loss_2[ind_2_sorted]
#
#     remember_rate = 1 - forget_rate
#     num_remember = int(remember_rate * len(loss_1_sorted))
#
#     ind_1_update = ind_1_sorted[:num_remember]
#     ind_2_update = ind_2_sorted[:num_remember]
#     # exchange
#     loss_1_update = F.cross_entropy(y_1[ind_2_update], t[ind_2_update])
#     loss_2_update = F.cross_entropy(y_2[ind_1_update], t[ind_1_update])
#
#     if torch.isnan(loss_1_update):
#         print(loss_1_update, y_1[ind_2_update], t[ind_2_update], len(loss_1_sorted), remember_rate)
#
#     return torch.sum(loss_1_update)/num_remember, torch.sum(loss_2_update)/num_remember
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def loss_coteaching(y_1, y_2, t, forget_rate, ind, noise_or_not):
    loss_1 = F.cross_entropy(y_1, t, reduce = False)
    ind_1_sorted = np.argsort(loss_1.data.cpu()).to(device)
    loss_1_sorted = loss_1[ind_1_sorted]

    loss_2 = F.cross_entropy(y_2, t, reduce = False)
    ind_2_sorted = np.argsort(loss_2.data.cpu()).to(device)
    loss_2_sorted = loss_2[ind_2_sorted]

    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_1_sorted))



    pure_ratio_1 = np.sum(noise_or_not[ind[ind_1_sorted.data.cpu()[:num_remember]]])/float(num_remember)
    pure_ratio_2 = np.sum(noise_or_not[ind[ind_2_sorted.data.cpu()[:num_remember]]])/float(num_remember)

    ind_1_update=ind_1_sorted[:num_remember]
    ind_2_update=ind_2_sorted[:num_remember]
    # exchange
    loss_1_update = F.cross_entropy(y_1[ind_2_update], t[ind_2_update])
    loss_2_update = F.cross_entropy(y_2[ind_1_update], t[ind_1_update])

    return torch.sum(loss_1_update)/num_remember, torch.sum(loss_2_update)/num_remember, pure_ratio_1, pure_ratio_2

