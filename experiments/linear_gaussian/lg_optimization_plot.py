
from models.backdoor import VariationalBackdoorWithSampling, VariationalBackdoor
from models.vae import GaussianIWAE
from models.nn import SimpleGaussianNN
from data.linear_gaussian import LinearGaussian
import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn.metrics import mean_absolute_error
import scipy.stats as st

dims = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

true_likelihood = []
separate_likelihood = {'low': [], 'high': [], 'mean': []}
finetuned_likelihood = {'low': [], 'high': [], 'mean': []}

separate_err = {'low': [], 'high': [], 'mean': []}
finetuned_err =  {'low': [], 'high': [], 'mean': []}


def confidence_interval(data, confidence=0.90):
    return st.norm.interval(confidence, loc=data.mean(), scale=st.sem(data))


for dim in dims:
    true_lst = []
    separate = []
    finetuned = []
    separate_mae = []
    finetuned_mae = []
    for seed in seeds:

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        data_size = 1000

        confounder = GaussianIWAE(feature_size=dim, latent_size=dim, class_size=0, hidden_size=10, num_samples=10).to(device)
        target = SimpleGaussianNN(feature_size=dim, class_size=2 * dim).to(device)
        encoder = SimpleGaussianNN(feature_size=dim, class_size=2 * dim).to(device)

        vb_separate = VariationalBackdoor(confounder_model=confounder, 
                                    target_model=target, 
                                    encoder_model=encoder, 
                                    backdoor_samples=10, 
                                    component_samples=10)

        vb_separate.load_state_dict(torch.load(f'trained_models/linear_gaussian/lg_dim={dim}_seed={seed}_no_joint.pt'))

        vb_separate.eval()

        encoder_finetuned = SimpleGaussianNN(feature_size=dim, class_size=2 * dim).to(device)

        vb_finetuned = VariationalBackdoor(confounder_model=confounder, 
                                    target_model=target, 
                                    encoder_model=encoder_finetuned, 
                                    backdoor_samples=10, 
                                    component_samples=10)

        vb_finetuned.load_state_dict(torch.load(f'trained_models/linear_gaussian/lg_dim={dim}_seed={seed}_finetuned.pt'))

        vb_finetuned.eval()



        dataset = LinearGaussian(data_size, dim, suppress_print=True, param_seed=seed, generation_seed=99)

        truth = dataset.ground_truth_do_likelihood(dataset.Y, dataset.X)
        truth = np.sum(np.log(truth), axis=-1)

        Y = torch.tensor(dataset.Y, dtype=torch.float32).to(device)
        X = torch.tensor(dataset.X, dtype=torch.float32).to(device)

        separate_est = vb_separate.get_log_backdoor(Y, X, 100, 5).detach().cpu().numpy()
        finetuned_est = vb_finetuned.get_log_backdoor(Y, X, 100, 5).detach().cpu().numpy()

        true_lst.append(truth.sum())
        separate.append(separate_est.sum())
        finetuned.append(finetuned_est.sum())
        separate_mae.append(mean_absolute_error(separate_est, truth))
        finetuned_mae.append(mean_absolute_error(finetuned_est, truth))

    true_lst = np.array(true_lst)
    separate = np.array(separate)
    finetuned = np.array(finetuned)
    separate_mae = np.array(separate_mae)
    finetuned_mae = np.array(finetuned_mae)
    
    true_likelihood.append(true_lst.mean())

    lower, upper = confidence_interval(separate)
    separate_likelihood['low'].append(lower)
    separate_likelihood['high'].append(upper)
    separate_likelihood['mean'].append(separate.mean())

    lower, upper = confidence_interval(finetuned)
    finetuned_likelihood['low'].append(lower)
    finetuned_likelihood['high'].append(upper)
    finetuned_likelihood['mean'].append(finetuned.mean())

    lower, upper = confidence_interval(separate_mae)
    separate_err['low'].append(lower)
    separate_err['high'].append(upper)
    separate_err['mean'].append(separate_mae.mean())

    lower, upper = confidence_interval(finetuned_mae)
    finetuned_err['low'].append(lower)
    finetuned_err['high'].append(upper)
    finetuned_err['mean'].append(finetuned_mae.mean())

plt.rcParams.update({'font.size': 14})

plt.plot(dims, finetuned_likelihood['mean'], label='Finetuning', color='royalblue')
plt.fill_between(dims, finetuned_likelihood['low'], finetuned_likelihood['high'], alpha=.5, color='royalblue', linewidth=0)
plt.plot(dims, separate_likelihood['mean'], label='Separate Training', color='limegreen')
plt.fill_between(dims, separate_likelihood['low'], separate_likelihood['high'], alpha=.5, color='limegreen', linewidth=0)
plt.plot(dims, true_likelihood, label='Ground Truth', color='purple')
plt.ylabel('Log Likelihood')
plt.xlabel('Dimension')
plt.legend()
plt.show()

plt.figure()
plt.plot(dims, finetuned_likelihood['mean'], label='Finetuning', color='royalblue')
plt.fill_between(dims, finetuned_likelihood['low'], finetuned_likelihood['high'], alpha=.5, color='royalblue', linewidth=0)
plt.plot(dims, separate_likelihood['mean'], label='Separate Training', color='limegreen')
plt.fill_between(dims, separate_likelihood['low'], separate_likelihood['high'], alpha=.5, color='limegreen', linewidth=0)
plt.plot(dims, true_likelihood, label='Ground Truth', color='purple')
plt.ylabel('Log Likelihood')
plt.yscale('symlog')
plt.xlabel('Dimension')
plt.legend()
plt.show()

plt.figure()
plt.plot(dims, finetuned_err['mean'], label='Finetuning', color='royalblue')
plt.fill_between(dims, finetuned_err['low'], finetuned_err['high'], alpha=.5, color='royalblue', linewidth=0)
plt.plot(dims, separate_err['mean'], label='Separate Training', color='limegreen')
plt.fill_between(dims, separate_err['low'], separate_err['high'], alpha=.5, color='limegreen', linewidth=0)

plt.ylabel('Mean Absolute Log Error')
plt.xlabel('Dimension')
plt.legend()
plt.show()

plt.figure()
plt.plot(dims, finetuned_err['mean'], label='Finetuning', color='royalblue')
plt.fill_between(dims, finetuned_err['low'], finetuned_err['high'], alpha=.5, color='royalblue', linewidth=0)
plt.plot(dims, separate_err['mean'], label='Separate Training', color='limegreen')
plt.fill_between(dims, separate_err['low'], separate_err['high'], alpha=.5, color='limegreen', linewidth=0)

plt.ylabel('Mean Absolute Log Error')
plt.yscale('log')
plt.xlabel('Dimension')
plt.legend()
plt.show()



