import torch
from torch import nn, optim
import torch.distributions as td
import numpy as np
import lightning.pytorch as pl

class BaseIWAE(nn.Module):
    def __init__(self, feature_size, latent_size, class_size, hidden_size, num_samples):
        super(BaseIWAE, self).__init__()
        self.feature_size = feature_size
        self.class_size = class_size
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.num_samples = num_samples
        
        # encode
        self.fc1  = nn.Linear(feature_size + class_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc31 = nn.Linear(hidden_size, latent_size)
        self.fc32 = nn.Linear(hidden_size, latent_size)

    def encode(self, x, c):
        if c is not None:
            inputs = torch.cat([x, c], dim=1)
        else:
            inputs = x

        h1 = self.relu(self.fc1(inputs))
        h2 = self.relu(self.fc2(h1))
        z_mu = self.fc31(h2)
        z_logvar = self.fc32(h2)
        return z_mu, z_logvar
        

    def decode(self, z, c): 
        pass

    def sample(self, num_samples, c):
        pass

    def forward(self, x, c, k):
        pass
    
    def compute_loss(self, x, c):
        log_w = self.get_importance_weights(x, c, self.num_samples)

        # log normalized trick
        log_w_tilde = log_w - torch.logsumexp(log_w, dim=1, keepdim=True)
        w_tilde = log_w_tilde.exp().detach()
        loss = -(w_tilde * log_w).sum(1).mean()
        return loss

    def step(self, opt, x, c):
        opt.zero_grad()
        loss = self.compute_loss(x, c)
        loss.backward()
        opt.step()
        return loss.detach().cpu().numpy()
    
    def get_importance_weights(self, x, c, k):
        pass

    def get_log_likelihood(self, x, c, k=None):
        if k is None:
            k = self.num_samples
        log_w = self.get_importance_weights(x, c, k)
        elbo = (torch.logsumexp(log_w, dim=1) -  np.log(k))
        return elbo
        

    def get_likelihood(self, x, c, k):
        return self.get_log_likelihood(x, c, k).exp()


class BernoulliIWAE(BaseIWAE):
    def __init__(self, feature_size, latent_size, class_size, hidden_size, num_samples):
        
        super(BernoulliIWAE, self).__init__(feature_size, latent_size, class_size, hidden_size, num_samples)

        # decode
        self.fc3 = nn.Linear(latent_size + class_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, feature_size)
        
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def decode(self, z, c):
        if c is not None:
            c = c.unsqueeze(1).repeat(1, z.size(1), 1)
            inputs = torch.cat([z, c], dim=-1)

        else:
            inputs = z
        h3 = self.relu(self.fc3(inputs))
        h4 = self.relu(self.fc4(h3))
        
        recon_x = self.sigmoid(self.fc5(h4))
        
        return recon_x
    
    def sample(self, num_samples, c):
        if c is None:
            batch_size = 1
        else:
            batch_size = c.size(0)
        random = torch.randn(batch_size, num_samples, self.latent_size).to('cuda')
        recon_x = self.decode(random, c)
        dist = td.Bernoulli(recon_x)
        samp = dist.sample()
        return samp

    def forward(self, x, c, k):
        mu_z, logvar_z = self.encode(x, c)

        std = torch.exp(0.5*logvar_z)
        
        q_z_g_x = td.Normal(loc=mu_z, scale=std)
        
        #samples using reparameterization trick
        z = q_z_g_x.rsample(torch.empty(k).size()).permute(1, 0, -1)

        recon_x = self.decode(z, c)
        return recon_x, mu_z, logvar_z, z, q_z_g_x
    
    def get_importance_weights(self, x, c, k):

        [recon_x, mu_z, logvar_z, z, q_z_g_x] = self.forward(x, c, k)
        
        
        x_s = x.unsqueeze(1).repeat(1, k, 1)
        
        mu_prior = torch.zeros(self.latent_size).to('cuda')
        std_prior = torch.ones(self.latent_size).to('cuda')
        p_z = td.Normal(loc=mu_prior, scale=std_prior)
        log_p_z = p_z.log_prob(z)
        
        
        log_p_x_g_z = td.Bernoulli(recon_x).log_prob(x_s)

        mu_z_s = mu_z.unsqueeze(1).repeat(1, k, 1)
        std_z_s = (0.5 * logvar_z).exp().unsqueeze(1).repeat(1, k, 1)
        log_q_z_g_x = td.Normal(mu_z_s, std_z_s).log_prob(z)


        log_p_z = log_p_z.sum(2)
        log_q_z_g_x = log_q_z_g_x.sum(2)
        log_p_x_g_z = log_p_x_g_z.sum(2)
        
        log_w = (log_p_x_g_z + log_p_z - log_q_z_g_x)
        
        return log_w
    

class GaussianIWAE(BaseIWAE):
    def __init__(self, feature_size, latent_size, class_size, hidden_size, num_samples):
        super(GaussianIWAE, self).__init__(feature_size, latent_size, class_size, hidden_size, num_samples)

        # decode
        self.fc3 = nn.Linear(latent_size + class_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc51 = nn.Linear(hidden_size, feature_size)
        self.fc52 = nn.Linear(hidden_size, feature_size)

        self.relu = nn.ReLU()

    def decode(self, z, c): 
        if c is not None:
            c = c.unsqueeze(1).repeat(1, z.size(1), 1)
            inputs = torch.cat([z, c], dim=-1)

        else:
            inputs = z

        h3 = self.relu(self.fc3(inputs))
        h4 = self.relu(self.fc4(h3))
        x_mu = self.fc51(h4)
        x_logvar = self.fc52(h4)
        return x_mu, x_logvar
    
    def sample(self, num_samples, c):
        if c is None:
            batch_size = 1
        else:
            batch_size = c.size(0)
        random = torch.randn(batch_size, num_samples, self.latent_size).to('cuda')
        recon_x, logvar_x = self.decode(random, c)
        std_x = torch.exp(0.5*logvar_x)
        dist = td.Normal(loc=recon_x, scale=std_x)
        samp = dist.sample()
        return samp

    def forward(self, x, c, k):
        mu_z, logvar_z = self.encode(x, c)

        std = torch.exp(0.5*logvar_z)
        q_z_g_x = td.Normal(loc=mu_z, scale=std)
        
        # samples using reparameterization trick
        z = q_z_g_x.rsample(torch.empty(k).size()).permute(1, 0, -1)
        recon_x, logvar_x = self.decode(z, c)
        return recon_x, logvar_x, mu_z, logvar_z, z, q_z_g_x
    
    def get_importance_weights(self, x, c, k):

        [recon_x, logvar_x, mu_z, logvar_z, z, q_z_g_x] = self.forward(x, c, k)
        
        x_s = x.unsqueeze(1).repeat(1, k, 1)
        
        mu_prior = torch.zeros(self.latent_size).to('cuda')
        std_prior = torch.ones(self.latent_size).to('cuda')
        p_z = td.Normal(loc=mu_prior, scale=std_prior)
        log_p_z = p_z.log_prob(z)

        eps = torch.finfo(torch.float32).eps
        
        std_x = (0.5 * logvar_x).exp() + eps

        p_x_g_z = td.Normal(loc=recon_x, scale=std_x)
        log_p_x_g_z = p_x_g_z.log_prob(x_s)

        mu_z_s = mu_z.unsqueeze(1).repeat(1, k, 1)
        std_z_s = (0.5 * logvar_z).exp().unsqueeze(1).repeat(1, k, 1)
        log_q_z_g_x = td.Normal(mu_z_s, std_z_s).log_prob(z)

        log_p_z = log_p_z.sum(2)
        log_q_z_g_x = log_q_z_g_x.sum(2)
        log_p_x_g_z = log_p_x_g_z.sum(2)

        log_w = (log_p_x_g_z + log_p_z - log_q_z_g_x)
        
        return log_w
    

class ImageGaussianIWAE(BaseIWAE):
    def __init__(self, feature_size, latent_size, class_size, hidden_size, num_samples):
        super(ImageGaussianIWAE, self).__init__(feature_size, latent_size, class_size, hidden_size, num_samples)

        # decode
        self.fc3 = nn.Linear(latent_size + class_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc51 = nn.Linear(hidden_size, feature_size)
        self.fc52 = nn.Linear(hidden_size, feature_size)

        self.relu = nn.ReLU()

    def decode(self, z, c): 
        if c is not None:
            c = c.unsqueeze(1).repeat(1, z.size(1), 1)
            inputs = torch.cat([z, c], dim=-1)

        else:
            inputs = z

        self.sigmoid = nn.Sigmoid()

        h3 = self.relu(self.fc3(inputs))
        h4 = self.relu(self.fc4(h3))
        x_mu = self.fc51(h4)
        x_logvar = self.fc52(h4)
        return self.sigmoid(x_mu), x_logvar
    
    def sample(self, num_samples, c):
        if c is None:
            batch_size = 1
        else:
            batch_size = c.size(0)
        random = torch.randn(batch_size, num_samples, self.latent_size).to('cuda')
        recon_x, logvar_x = self.decode(random, c)
        std_x = torch.exp(0.5*logvar_x)
        dist = td.Normal(loc=recon_x, scale=std_x)
        samp = dist.sample()
        return samp

    def forward(self, x, c, k):
        mu_z, logvar_z = self.encode(x, c)

        std = torch.exp(0.5*logvar_z)
        q_z_g_x = td.Normal(loc=mu_z, scale=std)
        
        #samples using reparameterization trick
        z = q_z_g_x.rsample(torch.empty(k).size()).permute(1, 0, -1)
        recon_x, logvar_x = self.decode(z, c)
        return recon_x, logvar_x, mu_z, logvar_z, z, q_z_g_x
    
    def get_importance_weights(self, x, c, k):

        [recon_x, logvar_x, mu_z, logvar_z, z, q_z_g_x] = self.forward(x, c, k)
        
        x_s = x.unsqueeze(1).repeat(1, k, 1)
        
        mu_prior = torch.zeros(self.latent_size).to('cuda')
        std_prior = torch.ones(self.latent_size).to('cuda')
        p_z = td.Normal(loc=mu_prior, scale=std_prior)
        log_p_z = p_z.log_prob(z)

        eps = torch.finfo(torch.float32).eps
        
        std_x = (0.5 * logvar_x).exp() + eps
        p_x_g_z = td.Normal(loc=recon_x, scale=std_x)
        log_p_x_g_z = p_x_g_z.log_prob(x_s)

        mu_z_s = mu_z.unsqueeze(1).repeat(1, k, 1)
        std_z_s = (0.5 * logvar_z).exp().unsqueeze(1).repeat(1, k, 1)
        log_q_z_g_x = td.Normal(mu_z_s, std_z_s).log_prob(z)

        log_p_z = log_p_z.sum(2)
        log_q_z_g_x = log_q_z_g_x.sum(2)
        log_p_x_g_z = log_p_x_g_z.sum(2)

        log_w = (log_p_x_g_z + log_p_z - log_q_z_g_x)
        
        return log_w

class SimpleGaussianIWAE(GaussianIWAE):
    def __init__(self, feature_size, latent_size, class_size, num_samples):
        super(GaussianIWAE, self).__init__(feature_size, latent_size, class_size, 1, num_samples)

        self.num_samples = num_samples

        # encode
        self.fc11  = nn.Linear(feature_size + class_size, latent_size)
        self.fc12  = nn.Linear(feature_size + class_size, latent_size)

        # decode
        self.fc21 = nn.Linear(latent_size + class_size, feature_size)
        self.fc22 = nn.Linear(latent_size + class_size, feature_size)



    def encode(self, x, c):
        if c is not None:
            inputs = torch.cat([x, c], dim=1)
        else:
            inputs = x

        z_mu = self.fc11(inputs)
        z_logvar = self.fc12(inputs)
        return z_mu, z_logvar

    def decode(self, z, c): 
        if c is not None:
            c = c.unsqueeze(1).repeat(1, z.size(1), 1)
            inputs = torch.cat([z, c], dim=-1)

        else:
            inputs = z

        x_mu = self.fc21(inputs)
        x_logvar = self.fc22(inputs)
        return x_mu, x_logvar
    
    

class CategoricalIWAE(BaseIWAE):

    def __init__(self, categories, latent_size, class_size, hidden_size, num_samples):
        super(CategoricalIWAE, self).__init__(sum(categories), latent_size, class_size, hidden_size, num_samples)

        self.feature_size = sum(categories)
        self.categories = categories
        # decode
        self.fc3 = nn.Linear(latent_size + class_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.ModuleList([nn.Linear(hidden_size, n) for n in self.categories])
        self.relu = nn.ReLU()


    def decode(self, z, c): 
        if c is not None:
            c = c.unsqueeze(1).repeat(1, z.size(1), 1)
            inputs = torch.cat([z, c], dim=-1)

        else:
            inputs = z

        h3 = self.relu(self.fc3(inputs))
        h4 = self.relu(self.fc4(h3))

        logits = [fc(h4) for i, fc in enumerate(self.fc5)]

        probs = [nn.functional.softmax(logit, dim=1) for logit in logits]

        return probs

    
    def sample(self, num_samples, c):
        if c is None:
            batch_size = 1
        else:
            batch_size = c.size(0)
        random = torch.randn(batch_size, num_samples, self.latent_size).to('cuda')
        probs = self.decode(random, c)
        samples = []
        for i, prob in enumerate(probs):
            dist = td.Categorical(probs=prob)
            index = dist.sample()
            sample = torch.eye(self.categories[i]).to('cuda')
            samples.append(sample[index])
        result = torch.cat(samples, dim=2)

        return result

    def forward(self, x, c, k):
        mu_z, logvar_z = self.encode(x, c)

        std = torch.exp(0.5*logvar_z)
        q_z_g_x = td.Normal(loc=mu_z, scale=std)
        
        #samples using reparameterization trick
        z = q_z_g_x.rsample(torch.empty(k).size()).permute(1, 0, -1)
        probs = self.decode(z, c)
        return probs, mu_z, logvar_z, z, q_z_g_x
    
    def get_importance_weights(self, x, c, k):

        [probs, mu_z, logvar_z, z, q_z_g_x] = self.forward(x, c, k)
        
        x_s = x.unsqueeze(1).repeat(1, k, 1)
        split_sizes = list(self.categories)
        x_s = torch.split(x_s, split_sizes, dim=2)

    
        
        mu_prior = torch.zeros(self.latent_size).to('cuda')
        std_prior = torch.ones(self.latent_size).to('cuda')
        p_z = td.Normal(loc=mu_prior, scale=std_prior)
        log_p_z = p_z.log_prob(z)
        

        log_p_x_g_z = torch.zeros((log_p_z.size(0), log_p_z.size(1))).to('cuda')

        for i, prob in enumerate(probs):
            p_x_g_z = td.Categorical(probs=prob)
            x = torch.argmax(x_s[i], dim=2)
            log_p_x_g_z += (p_x_g_z.log_prob(x))

        mu_z_s = mu_z.unsqueeze(1).repeat(1, k, 1)
        std_z_s = (0.5 * logvar_z).exp().unsqueeze(1).repeat(1, k, 1)
        log_q_z_g_x = td.Normal(mu_z_s, std_z_s).log_prob(z)

        log_p_z = log_p_z.sum(2)
        log_q_z_g_x = log_q_z_g_x.sum(2)
        log_p_x_g_z = log_p_x_g_z

        log_w = (log_p_x_g_z + log_p_z - log_q_z_g_x)
        
        return log_w

class ConditionalIWAETrainingWrapper(pl.LightningModule):
    
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def training_step(self, batch, batch_idx):
        X, Y, Z = batch

        loss = self.model.compute_loss(Y, X)
 
        self.log('loss', loss)

        return loss
    
    def configure_optimizers(self):
        opt =  optim.Adam(self.model.parameters(), lr=1e-3)
        return opt
    

class IWAETrainingWrapper(pl.LightningModule):
    
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def training_step(self, batch, batch_idx):
        X, Y, Z = batch

        loss = self.model.compute_loss(Z, None)
 
        self.log('loss', loss)

        return loss
    
    def configure_optimizers(self):
        opt =  optim.Adam(self.model.parameters(), lr=1e-3)
        return opt
    
class ConditionalIWAETrainingWrapper(pl.LightningModule):
    
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def training_step(self, batch, batch_idx):
        X, Y, Z = batch

        loss = self.model.compute_loss(Y, X)
 
        self.log('loss', loss)

        return loss
    
    def configure_optimizers(self):
        opt =  optim.Adam(self.model.parameters(), lr=1e-3)
        return opt

