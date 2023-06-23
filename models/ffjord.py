


import torch

# custom architectures from separate ffjord library based on default MNIST training
from ffjord.load_mnist import load_ffjord_mnist
from ffjord.load_xray import load_ffjord_xray

from torch import nn
import math

def standard_normal_logprob(z):
    logZ = -0.5 * math.log(2 * math.pi)
    return logZ - z.pow(2) / 2

class FFJORD(nn.Module):
    def __init__(self, data):

        assert data == 'mnist' or data == 'xray'
        super(FFJORD, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if data == 'mnist':
            model, train_loader = load_ffjord_mnist()
        else:
            model, train_loader = load_ffjord_xray()
        self.model = model

    def forward(self):
        pass

    def get_log_likelihood(self, x, c, k):
        x = torch.clamp(x.view((-1, 1, 28, 28)), min=0, max=1)
        zero = torch.zeros(x.size(0), 1).to(self.device)

        z, delta_logp = self.model(x, zero)

        logpz = standard_normal_logprob(z).view(z.size(0), -1).sum(1, keepdim=True)

        log_likelihood = logpz - delta_logp

        return log_likelihood.squeeze()

    
    def compute_loss(self, x, c):
        return -self.get_log_likelihood(x, c).mean()
        
    def sample(self, num_samples, c):
        data_shape = (1, 28, 28)
        z = torch.randn(num_samples, *data_shape).to('cuda')
        generated_samples = self.model(z, reverse=True).view(1, num_samples, 28 * 28)
        return generated_samples

