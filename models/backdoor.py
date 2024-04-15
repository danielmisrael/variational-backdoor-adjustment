import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.utils.data import TensorDataset
import lightning.pytorch as pl

import matplotlib.pyplot as plt

class VariationalBackdoor(nn.Module):
    
    def __init__(self, confounder_model, target_model, encoder_model, backdoor_samples, component_samples):
        super(VariationalBackdoor, self).__init__()
        self.backdoor_samples = backdoor_samples
        self.component_samples = component_samples
        
        self.confounder = confounder_model
        self.target = target_model
        self.encoder = encoder_model
    
    def get_log_backdoor(self, Y, X, Z_prime, backdoor_samples=100, component_samples=100):
        
        batch_size = 64
        result = []
        
        Y = torch.split(Y, batch_size)
        X = torch.split(X, batch_size)
        Z_prime = torch.split(Z_prime, batch_size)

        for x, y, z_prime in zip(X, Y, Z_prime):

            batch_size = x.size(0)

            z = self.encoder.sample(backdoor_samples, torch.cat([x, y, z_prime], 1))
            z = z.flatten(start_dim=0, end_dim=1)
            x = x.unsqueeze(1).repeat(1, backdoor_samples, 1).flatten(start_dim=0, end_dim=1)
            y = y.unsqueeze(1).repeat(1, backdoor_samples, 1).flatten(start_dim=0, end_dim=1)
            z_prime = z_prime.unsqueeze(1).repeat(1, backdoor_samples, 1).flatten(start_dim=0, end_dim=1)

            log_p_z = self.confounder.get_log_likelihood(z, z_prime, k=component_samples)
            log_p_y_xz = self.target.get_log_likelihood(y, torch.cat([x, z, z_prime], 1))
            log_q_z_xy = self.encoder.get_log_likelihood(z, torch.cat([x, y, z_prime], 1))

            log_bw =  log_p_z + log_p_y_xz - log_q_z_xy
            log_bw = log_bw.reshape(batch_size, backdoor_samples)

            backdoor_elbo = (torch.logsumexp(log_bw, dim=1) -  np.log(backdoor_samples))

            result.append(backdoor_elbo)

        return torch.cat(result)
    
    def compute_backdoor_loss(self, Y, X, Z_prime, backdoor_samples=100, component_samples=100):

        batch_size = 64
        result = []
        
        Y = torch.split(Y, batch_size)
        X = torch.split(X, batch_size)
        #z_prime
        Z_prime = torch.split(Z_prime, batch_size)


        for x, y, z_prime in zip(X, Y, Z_prime):

            batch_size = x.size(0)

            z = self.encoder.sample(backdoor_samples, torch.cat([x, y], 1))
            z = z.flatten(start_dim=0, end_dim=1)
            x = x.unsqueeze(1).repeat(1, backdoor_samples, 1).flatten(start_dim=0, end_dim=1)
            y = y.unsqueeze(1).repeat(1, backdoor_samples, 1).flatten(start_dim=0, end_dim=1)

            #Send z_prime to the confounder model
            log_p_z = self.confounder.get_log_likelihood(z, z_prime, k=component_samples)
            log_p_y_xz = self.target.get_log_likelihood(y, torch.cat([x, z, z_prime], 1))

            #TO DO: need to send z_prime to the encoder model by concatenation
            log_q_z_xy = self.encoder.get_log_likelihood(z, torch.cat([x, y, z_prime], 1))

            log_bw =  log_p_z + log_p_y_xz - log_q_z_xy
            log_bw = log_bw.reshape(batch_size, backdoor_samples)
            result.append(log_bw)

        log_bw = torch.cat(result)

        # numerical stability 
        log_w_minus_max = log_bw - log_bw.max(1, keepdim=True)[0]
        # compute normalized importance weights (no gradient)
        w = log_w_minus_max.exp()
        w_tilde = (w / w.sum(axis=1, keepdim=True)).detach()
        # compute loss (negative IWAE objective)
        loss = -(w_tilde * log_bw).sum(1).mean()
        return loss
    
    def get_backdoor(self, Y, X, backdoor_samples, component_samples):
        return self.get_log_backdoor(Y, X, backdoor_samples, component_samples).exp()
    
    def sample(self, num_samples, do_X):
        
        batch_size = do_X.size(0)
        
        
        new_Z = self.confounder.sample(num_samples * batch_size, None)
        new_Z = new_Z.flatten(start_dim=0, end_dim=1)
        
        do_X = do_X.unsqueeze(dim=1).repeat(1, num_samples, 1).flatten(start_dim=0, end_dim=1)
        
        new_Y = self.target.sample(1, torch.cat([do_X, new_Z], 1))
        new_Y = new_Y.flatten(start_dim=0, end_dim=1)
        
        return new_Y
    

class SeparateTrainingWrapper(pl.LightningModule):
    
    def __init__(self, vb, ignore_confounder=False):
        super().__init__()
        self.vb = vb
        self.ignore_confounder = ignore_confounder
    
    def training_step(self, batch, batch_idx):
        X, Y, Z, Z_prime = batch      #get z_prime as well  

        target_loss = self.vb.target.compute_loss(Y, torch.cat([X, Z, Z_prime], 1))
        self.log('target_loss', target_loss)

        encoder_loss = self.vb.encoder.compute_loss(Z, torch.cat([X, Y, Z_prime], 1))
        self.log('encoder_loss', encoder_loss)

        if not self.ignore_confounder:

            confounder_loss = self.vb.confounder.compute_loss(Z, Z_prime)
            self.log('confounder_loss', target_loss)
            return target_loss + confounder_loss + encoder_loss
        
        else:
            return target_loss + encoder_loss

    def configure_optimizers(self):
        opt =  optim.Adam(self.vb.parameters(), lr=1e-3)
        return opt
    

class FinetuningWrapper(pl.LightningModule):
    
    def __init__(self, vb, lr=1e-3):
        super().__init__()
        self.vb = vb
        self.lr = lr
    
    def training_step(self, batch, batch_idx):
        X, Y, Z, Z_prime = batch

        backdoor_loss = -self.vb.get_log_backdoor(Y, X, Z_prime, backdoor_samples=self.vb.backdoor_samples,
                    component_samples=self.vb.component_samples).mean()
        

        self.log('backdoor_loss', backdoor_loss)

        return backdoor_loss

    def configure_optimizers(self):
        opt =  optim.Adam(self.vb.encoder.parameters(), lr=self.lr)
        return opt
    

class FullyJointFinetuningWrapper(pl.LightningModule):
    
    def __init__(self, vb, lr=1e-3):
        super().__init__()
        self.vb = vb
        self.lr = lr
    
    def training_step(self, batch, batch_idx):
        X, Y, Z = batch

        backdoor_loss = -self.vb.get_log_backdoor(Y, X, backdoor_samples=self.vb.backdoor_samples,
                    component_samples=self.vb.component_samples).mean()
        
        self.log('backdoor_loss', backdoor_loss)

        return backdoor_loss

    def configure_optimizers(self):
        opt =  optim.Adam(self.vb.parameters(), lr=self.lr)
        return opt
    

class VariationalBackdoorWithSampling(VariationalBackdoor):

    def __init__(self, confounder_model, target_model, encoder_model, backdoor_samples, component_samples):
        super(VariationalBackdoorWithSampling, self).__init__(confounder_model, target_model, encoder_model,
                                                               backdoor_samples, component_samples)
    
    def get_log_backdoor_sampling(self, Y, X, num_samples=100):

        batch_size = 64
        result = []
        
        Y = torch.split(Y, batch_size)
        X = torch.split(X, batch_size)

        for x, y in zip(X, Y):
        
            batch_size = x.size(0)

            z = self.confounder.sample(batch_size * num_samples, None)
            z = z.flatten(start_dim=0, end_dim=1)

            x = x.unsqueeze(1).repeat(1, num_samples, 1).flatten(start_dim=0, end_dim=1)
            y = y.unsqueeze(1).repeat(1, num_samples, 1).flatten(start_dim=0, end_dim=1)

            log_p_y_xz = self.target.get_log_likelihood(y, torch.cat([x, z], 1))
            log_p_y_xz = log_p_y_xz.reshape(batch_size, num_samples)

            result.append(torch.logsumexp(log_p_y_xz, dim=1) -  np.log(num_samples))

        return torch.cat(result)
    

    def get_backdoor_sampling(self, Y, X, num_samples=100):
        
        return self.get_log_backdoor_sampling(Y, X, num_samples).exp()
    

class VariationalBackdoorWithStats(VariationalBackdoor):

    def __init__(self, confounder_model, target_model, encoder_model, backdoor_samples, component_samples):
        super(VariationalBackdoorWithStats, self).__init__(confounder_model, target_model, encoder_model,
                                                               backdoor_samples, component_samples)
    
    def backdoor_stats(self, Y, X, backdoor_samples=100, component_samples=100):

        batch_size = Y.size(0)

        Z = self.encoder.sample(backdoor_samples, torch.cat([X, Y], 1))
        Z = Z.flatten(start_dim=0, end_dim=1)
        X = X.unsqueeze(1).repeat(1, backdoor_samples, 1).flatten(start_dim=0, end_dim=1)
        Y = Y.unsqueeze(1).repeat(1, backdoor_samples, 1).flatten(start_dim=0, end_dim=1)

        log_p_z = self.confounder.get_log_likelihood(Z, None, k=component_samples)
        log_p_y_xz = self.target.get_log_likelihood(Y, torch.cat([X, Z], 1))
        log_q_z_xy = self.encoder.get_log_likelihood(Z, torch.cat([X, Y], 1))


        log_bw =  log_p_z + log_p_y_xz - log_q_z_xy
        log_bw = log_bw.reshape(batch_size, backdoor_samples)

        backdoor_elbo = (torch.logsumexp(log_bw, dim=1) -  np.log(backdoor_samples))

        # average out each component over samples

        log_p_z = log_p_z.reshape(batch_size, backdoor_samples)
        log_p_z = (torch.logsumexp(log_p_z, dim=1) -  np.log(backdoor_samples))

        log_p_y_xz = log_p_y_xz.reshape(batch_size, backdoor_samples)
        log_p_y_xz = (torch.logsumexp(log_p_y_xz, dim=1) -  np.log(backdoor_samples))

        log_q_z_xy = log_q_z_xy.reshape(batch_size, backdoor_samples)
        log_q_z_xy = (torch.logsumexp(log_q_z_xy, dim=1) -  np.log(backdoor_samples))
        
        return backdoor_elbo.cpu().numpy(), log_p_z.cpu().numpy(), log_p_y_xz.cpu().numpy(), log_q_z_xy.cpu().numpy()
    
    
