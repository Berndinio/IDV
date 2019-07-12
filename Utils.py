
import torch


class Utils:
    g_device = None



if torch.cuda.is_available():
    Utils.g_device = torch.device('cuda')
else:
    Utils.g_device = torch.device('cpu')
