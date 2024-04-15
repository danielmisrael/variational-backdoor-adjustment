

from models.backdoor import VariationalBackdoor, SeparateTrainingWrapper, FinetuningWrapper
from models.vae import GaussianIWAE
from models.nn import SimpleGaussianNN 
import torch
import lightning.pytorch as pl
from lightning.pytorch.loggers import CSVLogger
import numpy as np
from torch.utils.data import Dataset

# Define some data
class SimpleExample(Dataset):
    def __init__(self, size, dim):
        np.random.seed(0)
        Z = np.random.normal(size=(size, dim))
        X = np.random.normal(size=(size, dim)) * Z
        Y = (X + Z) * np.random.normal(size=(size, dim))

        #create mask and z_prime
        mask = np.random.binomial(1, 0.5, size=(size, dim))
        print("mask", mask)
        Z_prime = torch.masked_fill(torch.tensor(Z).float(), torch.tensor(mask).bool(), 0)

        print(Z.shape)
        print(X.shape)
        print(Y.shape)

        self.Z = np.array(Z, dtype='float32')
        self.X = np.array(X, dtype='float32')
        self.Y = np.array(Y, dtype='float32')
        self.Z_prime = np.array(Z_prime, dtype='float32')

    def __len__(self):
        return len(self.Z)

    def __getitem__(self, n):
        # Tuple must be of the form X, Y, Z where
        # Z is the confounder (Z -> X and Z -> Y)
        # X is the treatment (X -> Y)
        # Y is the target
        return self.X[n], self.Y[n], self.Z[n], self.Z_prime[n]

torch.manual_seed(0)

dim = 10
data_size = 50000
train = SimpleExample(data_size, dim)
train_loader = torch.utils.data.DataLoader(train, batch_size=64, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Parameterize each of the following components with various models that can estimate log-likelihood each of the following distributions
# A variety of different models are given in the "models" folder. The best parameterization depends on the data.

# P(Z)
confounder = GaussianIWAE(feature_size=dim, latent_size=dim, class_size=dim, hidden_size=10, num_samples=5).to(device)

# P(Y | X, Z)
target = SimpleGaussianNN(feature_size=dim, class_size=3 * dim).to(device)

# P(Z | X, Y)
encoder = SimpleGaussianNN(feature_size=dim, class_size=3 * dim).to(device)

# Intialize class for Variational Backdoor Adjustment
vb = VariationalBackdoor(confounder_model=confounder, 
                            target_model=target, 
                            encoder_model=encoder, 
                            backdoor_samples=10, # Number of samples used to compute backdoor adjustment
                            component_samples=10) # Number of samples used to compute log-likelihood for the inner model


# Two step process: first train each of the components separately, then finetune the encoder to maximize the interventional density
print('Separate Component Training')
logger1 = CSVLogger('trained_models/example/logs', name='separate_training')
trainer1 = pl.Trainer(max_epochs=2, default_root_dir='trained_models/', logger=logger1)
trainer1.fit(model=SeparateTrainingWrapper(vb), train_dataloaders=train_loader)
torch.save(vb.state_dict(), f'trained_models/example/example.pt')

vb.load_state_dict(torch.load(f'trained_models/example/example.pt'))
print('Encoder Finetuning')
logger2 = CSVLogger('trained_models/example/logs', name='finetuning')
trainer2 = pl.Trainer(max_epochs=2, default_root_dir='trained_models/', logger=logger2)
trainer2.fit(model=FinetuningWrapper(vb), train_dataloaders=train_loader)
torch.save(vb.state_dict(), f'trained_models/example/example.pt')


vb.load_state_dict(torch.load(f'trained_models/example/example.pt'))
# Some test data from the same distribution
test = SimpleExample(1000, dim)
test_loader = torch.utils.data.DataLoader(test, batch_size=1000, shuffle=False)

for X, Y, Z in test_loader:
    # To obtain interventional likelihood estimate, call get_log_backdoor()
    Z_prime = 0
    interventional_likelihood = vb.get_log_backdoor(Y.to(device), X.to(device), Z_prime.to(device), backdoor_samples=10, component_samples=10)


print('The interventional log-likelihood of the test set is ', interventional_likelihood.sum().item())
