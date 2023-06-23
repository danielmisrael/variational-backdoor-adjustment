

import torch
from torch import nn
import torch.distributions as td
import torch.nn.functional as F
from torch.autograd import Variable


class BernoulliNN(nn.Module):
    def __init__(self, feature_size, class_size, hidden_size):
        super(BernoulliNN, self).__init__()
        self.feature_size = feature_size
        self.hidden_size = hidden_size

        # encode
        self.fc1  = nn.Linear(class_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, feature_size)


        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, c):

        h1 = self.relu(self.fc1(c))
        h2 = self.relu(self.fc2(h1))
        p = self.sigmoid(self.fc3(h2))
        
        return p


    def compute_loss(self, x, c):
        return -self.get_log_likelihood(x, c).mean()
    
    def step(self, opt, x, c):
        opt.zero_grad()
        loss = self.compute_loss(x, c)
        loss.backward()
        opt.step()
        return loss.detach().cpu().numpy()

    def get_log_likelihood(self, x, c):
        p = self.forward(c)
        dist = td.Bernoulli(p)
        prob = dist.log_prob(x)
        return prob.sum(-1)
    
    def sample(self, num_samples, c):
        p = self.forward(c)
        dist =  td.Bernoulli(probs=p)
        samp = dist.sample(torch.empty(num_samples).size())
        reparam = (samp - p).detach() + p
        return reparam.permute(1, 0, -1)
    


class GaussianNN(nn.Module):
    def __init__(self, feature_size, class_size, hidden_size):
        super(GaussianNN, self).__init__()
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        
        # encode
        self.fc1  = nn.Linear(class_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc31 = nn.Linear(hidden_size, feature_size)
        self.fc32 = nn.Linear(hidden_size, feature_size)

        self.relu = nn.ReLU()

    def forward(self, c):
        
        h1 = self.relu(self.fc1(c))
        h2 = self.relu(self.fc2(h1))
        mu = self.fc31(h2)
        logvar = self.fc32(h2)
        return mu, logvar

    def compute_loss(self, x, c):
        
        return -self.get_log_likelihood(x, c).mean()
    
    def step(self, opt, x, c):
        opt.zero_grad()
        loss = self.compute_loss(x, c)
        loss.backward()
        opt.step()
        return loss.detach().cpu().numpy()

    def get_log_likelihood(self, x, c):
        
        mu, logvar = self.forward(c)
        std = (0.5 * logvar).exp()
        normal_dist = td.Normal(loc=mu, scale=std)
        prob = normal_dist.log_prob(x)
        return prob.sum(-1)
    

    def sample(self, num_samples, c):

        mu, logvar = self.forward(c)

        std = torch.exp(0.5*logvar)
        dist = td.Normal(loc=mu, scale=std)
        samp = dist.rsample(torch.empty(num_samples).size())

        return samp.permute(1, 0, -1)
    

class ImageGaussianNN(nn.Module):
    def __init__(self, feature_size, class_size, hidden_size):
        super(ImageGaussianNN, self).__init__()
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        
        # encode
        self.fc1  = nn.Linear(class_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc31 = nn.Linear(hidden_size, feature_size)
        self.fc32 = nn.Linear(hidden_size, feature_size)

        self.relu = nn.ReLU()

    def forward(self, c):
        
        h1 = self.relu(self.fc1(c))
        h2 = self.relu(self.fc2(h1))
        mu = self.fc31(h2)
        logvar = self.fc32(h2)
        self.sigmoid = nn.Sigmoid()
        return self.sigmoid(mu), logvar

    def compute_loss(self, x, c):
        
        return -self.get_log_likelihood(x, c).mean()
    
    def step(self, opt, x, c):
        opt.zero_grad()
        loss = self.compute_loss(x, c)
        loss.backward()
        opt.step()
        return loss.detach().cpu().numpy()

    def get_log_likelihood(self, x, c):
        
        mu, logvar = self.forward(c)
        std = (0.5 * logvar).exp()
        normal_dist = td.Normal(loc=mu, scale=std)
        prob = normal_dist.log_prob(x)
        return prob.sum(-1)
    

    def sample(self, num_samples, c):

        mu, logvar = self.forward(c)
        std = torch.exp(0.5*logvar)
        dist = td.Normal(loc=mu, scale=std)
        samp = dist.rsample(torch.empty(num_samples).size())
        return samp.permute(1, 0, -1)
    

class CategoricalNN(nn.Module):
    def __init__(self, categories, class_size, hidden_size):
        super(CategoricalNN, self).__init__()
        self.feature_size = sum(categories)
        self.categories = categories
        self.hidden_size = hidden_size

        # encode
        self.fc1  = nn.Linear(class_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.ModuleList([nn.Linear(hidden_size, n) for n in self.categories])


        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, c):

        h1 = self.relu(self.fc1(c))
        h2 = self.relu(self.fc2(h1))

        logits = [fc(h2) for i, fc in enumerate(self.fc3)]

        probs = [nn.functional.softmax(logit, dim=1) for logit in logits]
        return logits, probs



    def compute_loss(self, x, c):
        return -self.get_log_likelihood(x, c).mean()
    
    def step(self, opt, x, c):
        opt.zero_grad()
        loss = self.compute_loss(x, c)
        loss.backward()
        opt.step()
        return loss.detach().cpu().numpy()

    def get_log_likelihood(self, x, c):

        split_sizes = list(self.categories)
        x_s = torch.split(x, split_sizes, dim=1)

        logits, probs = self.forward(c)

        log_likelihood = torch.zeros(x.size(0)).to('cuda')

        for i, prob in enumerate(probs):
            dist = td.Categorical(probs=prob)
            categorical = torch.argmax(x_s[i], dim=1)
            log_likelihood += (dist.log_prob(categorical))

        return log_likelihood

    
    def sample(self, num_samples, c):

        logits, probs = self.forward(c)


        samples = []
        for i, logit in enumerate(logits):
            logit = logit.unsqueeze(1).repeat(1, num_samples, 1)
            sample = gumbel_softmax(logit)

            samples.append(sample)
        result = torch.cat(samples, dim=2)

        return result
    



def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape).to('cuda')
    return -Variable(torch.log(-torch.log(U + eps) + eps))

def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits, temperature=1.0):
    """
    input: [*, n_class]
    return: [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)
    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    return (y_hard - y).detach() + y

"""
Alternative to Gumbel-Softmax
Ahmed, Kareem, et al. "SIMPLE: A Gradient Estimator for $ k $-Subset Sampling." arXiv preprint arXiv:2210.01941 (2022).
"""
def SIMPLE(logits):
    """
    input: [*, n_class]
    return: [*, n_class] an one-hot vector
    """
    y = F.softmax(logits, dim=-1)
    y_perturbed = F.softmax(logits + sample_gumbel(logits.size()), dim=-1)
    shape = y.size()
    _, ind = y_perturbed.max(dim=-1)
    y_hard = torch.zeros_like(y_perturbed).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    return (y_hard - y).detach() + y



class SimpleGaussianNN(GaussianNN):
    def __init__(self, feature_size, class_size):
        super(GaussianNN, self).__init__()
        self.feature_size = feature_size
        self.fc11 = nn.Linear(class_size, feature_size)
        self.fc12 = nn.Linear(class_size, feature_size)

    def forward(self, c):
        mu = self.fc11(c)
        logvar = self.fc12(c)

        return mu, logvar

