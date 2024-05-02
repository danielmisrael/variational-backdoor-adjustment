

from models.backdoor import VariationalBackdoor, SeparateTrainingWrapper, FinetuningWrapper
from models.vae import GaussianIWAE
from models.nn import GaussianNN 
import torch
import lightning.pytorch as pl
from lightning.pytorch.loggers import CSVLogger
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from scipy.special import logsumexp


def convert_one_hot(data):
    data_info = []
    cat_cols = []
    bin_cols = []
    num_cols = []
    for key in data:
        if 'HCC' not in key:
            data_type = df.loc[df['Feature Name'] == key]['Type'].values[0]
            if data_type == 'Categorical':
                cat_cols.append(key)
                data_info.append((data_type, len(data[key].dropna().unique())))
            elif data_type == 'Binary':
                bin_cols.append(key)
                data_info.append(('Binary', 1))
            else:
                num_cols.append(key)
                data_info.append(('Numerical', 1))
            
        else:
            
            bin_cols.append(key)
            data_info.append(('Binary', 1))

    data = data.reindex(bin_cols + cat_cols + num_cols, axis=1)

    data_info = sorted(data_info, key=lambda x:x[0])

    data = pd.get_dummies(data, columns=cat_cols)
    data = pd.get_dummies(data, columns=bin_cols, drop_first=True)

    return data, data_info

class PatientData(Dataset):
    def __init__(self, data):

        Z = data.iloc[:, 2:90]
        X = data.iloc[:, 90:113]
        Y = data.iloc[:, 113:]

        Z, Z_info = convert_one_hot(Z)
        X, X_info = convert_one_hot(X)
        Y, Y_info = convert_one_hot(Y)

        Z = Z.to_numpy().astype('float32')
        X = X.to_numpy().astype('float32')
        Y = Y.to_numpy().astype('float32')

        # Can be removed if no nan's
        Z = np.nan_to_num(Z)
        X = np.nan_to_num(X)
        Y = np.nan_to_num(Y)

        self.Z = Z
        self.X = X
        self.Y = Y


    def __len__(self):
        return len(self.Z)

    def __getitem__(self, n):
        # Tuple must be of the form X, Y, Z where
        # Z is the confounder (Z -> X and Z -> Y)
        # X is the treatment (X -> Y)
        # Y is the target
        # Z_prime is masked Z

        #create mask and z_prime
        mask = np.random.binomial(1, 0.5, size=(self.Z[n].shape))
        Z_prime = torch.masked_fill(torch.tensor(self.Z[n]).float(), torch.tensor(mask).bool(), 0)

        return self.X[n], self.Y[n], self.Z[n], torch.tensor(Z_prime)


torch.manual_seed(0)

# TODO update csv file with real data
data = pd.read_csv('dummy_data.csv')
df = pd.read_excel('data_info.xlsx')

train = PatientData(data)
train_loader = torch.utils.data.DataLoader(train, batch_size=4, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dim_X = len(train.X[0])
dim_Y = len(train.Y[0])
dim_Z = len(train.Z[0])

# # Parameterize each of the following components with various models that can estimate log-likelihood each of the following distributions
# # A variety of different models are given in the "models" folder. The best parameterization depends on the data.

# P(Z)
confounder = GaussianIWAE(feature_size=dim_Z, latent_size=10, class_size=dim_Z, hidden_size=200, num_samples=5).to(device)

# P(Y | X, Z)
target = GaussianNN(feature_size=dim_Y, class_size=dim_X+ 2 * dim_Z, hidden_size=200).to(device)

# P(Z | X, Y)
encoder = GaussianNN(feature_size=dim_Z, class_size=dim_X+dim_Y+dim_Z, hidden_size=200).to(device)

# Intialize class for Variational Backdoor Adjustment
vb = VariationalBackdoor(confounder_model=confounder, 
                            target_model=target, 
                            encoder_model=encoder, 
                            backdoor_samples=10, # Number of samples used to compute backdoor adjustment
                            component_samples=10) # Number of samples used to compute log-likelihood for the inner model


# Just doing separate component training for now
print('Separate Component Training')
logger1 = CSVLogger('trained_models/example/logs', name='separate_training')
trainer1 = pl.Trainer(max_epochs=1, default_root_dir='trained_models/', logger=logger1)
trainer1.fit(model=SeparateTrainingWrapper(vb), train_dataloaders=train_loader)
torch.save(vb.state_dict(), f'trained_models/example/patient_model.pt')


# This code below can be used for a recommender system
torch.save(vb.state_dict(), f'trained_models/example/patient_model.pt')

# Input masked Z aka Z_prime
Z_prime = torch.tensor(train.Z[0]).unsqueeze(0).to(device) # TODO change to input Z_prime

num_samples = 100
log_likelihoods = []
for _ in range(num_samples):
    # Get a random sample
    random_index = int(np.random.random()*len(train))
    X, Y, _, _ = train[random_index]

    Y = torch.tensor(Y).unsqueeze(0).to(device)
    X = torch.tensor(X).unsqueeze(0).to(device)

    with torch.no_grad():
        interventional_likelihoods = vb.get_log_backdoor(Y, X, Z_prime, backdoor_samples=10, component_samples=10)
    log_likelihoods.append(interventional_likelihoods.item())


log_likelihoods = np.array(log_likelihoods)
logsumexp_likelihood = logsumexp(log_likelihoods) - np.log(num_samples)
print("Backdoor Log Likelihood:", logsumexp_likelihood)


