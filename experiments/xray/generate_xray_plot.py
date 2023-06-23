from models.backdoor import VariationalBackdoor
from models.nn import ImageGaussianNN
from models.ffjord import FFJORD
import torch
import numpy as np
from data.xray import InterventionalXrayCausalDataset
import pickle

def bits_per_dim(log_likelihood, dim):

    logpx_per_dim = log_likelihood / dim
    bits_per_dim = -(logpx_per_dim - np.log(256)) / np.log(2)

    return bits_per_dim


def get_likelihood(data, model_fn):
    likelihood = 0
    batch_size = 100
    loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False)
    
    for i, (X, Y) in enumerate(loader):
        if i % 100 == 0:
            print('Progress', i,'/', len(loader))
        with torch.no_grad():
            likelihood += model_fn(Y.to(device), X.to(device)).sum()

    likelihood = likelihood / len(data)

    return likelihood
    

def save_results(vb_fn, data, name):

    likelihood = get_likelihood(data, vb_fn)

    bpd = bits_per_dim(likelihood, dim)

    results = {'log_likelihood': likelihood.item(), 'bpd': bpd.item()} 
    with open(name, 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


def save_accuracies(vb_fn, name):
    results = {}

    for acc in [0, 0.3, 0.5, 0.7, 1]:
        print(acc)

        data = InterventionalXrayCausalDataset(accuracy=acc)

        with torch.no_grad():
            likelihood = get_likelihood(data, vb_fn)

            bpd = bits_per_dim(likelihood, dim)

        results[acc] = (likelihood.item(), bpd.item())

    with open(name, 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

def save_bpd_data(model_fn, name):

    results = {}

    for acc in [0, 0.3, 0.5, 0.7, 1]:
        print(acc)

        data = InterventionalXrayCausalDataset(accuracy=acc)

        with torch.no_grad():

            likelihoods = np.array([])

            batch_size = 100
            loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False)
    
            for i, (X, Y) in enumerate(loader):
                if i % 100 == 0:
                    print('Progress', i,'/', len(loader))
                with torch.no_grad():
                    likelihood = model_fn(Y.to(device), X.to(device)).cpu().numpy()

                likelihoods = np.append(likelihoods, likelihood)

            print(likelihoods.shape)


            bpd = bits_per_dim(likelihoods, dim)

        results[acc] = bpd

    with open(name, 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


dim = 28 * 28
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



confounder = FFJORD(data='xray').to(device)
target = ImageGaussianNN(feature_size=dim, hidden_size=300, class_size=2 * dim).to(device)
encoder = ImageGaussianNN(feature_size=dim, hidden_size=300, class_size=2 * dim).to(device)

vb = VariationalBackdoor(confounder_model=confounder, 
                            target_model=target, 
                            encoder_model=encoder, 
                            backdoor_samples=1, 
                            component_samples=1)
vb.eval()
vb_fn = lambda y, x: vb.get_log_backdoor(y, x, 5, 1)
save_bpd_data(vb_fn, 'results/xray/xray_no_joint_accuracies_bpd.pickle')


vb.load_state_dict(torch.load('trained_models/xray/xray_tumor_vb_finetuned.pt'))
vb_fn = lambda y, x: vb.get_log_backdoor(y, x, 5, 1)
save_bpd_data(vb_fn, 'results/xray/xray_finetuned_accuracies_bpd.pickle')



