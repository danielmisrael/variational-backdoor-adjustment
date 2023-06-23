
from models.backdoor import VariationalBackdoorWithSampling
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
variational_likelihood = {'low': [], 'high': [], 'mean': []}
sampling_likelihood = {'low': [], 'high': [], 'mean': []}

variational_err = {'low': [], 'high': [], 'mean': []}
sampling_err =  {'low': [], 'high': [], 'mean': []}



def confidence_interval(data, confidence=0.90):
    return st.norm.interval(confidence, loc=data.mean(), scale=st.sem(data))

for dim in dims:
    t = []
    v = []
    s = []
    v_mae = []
    s_mae = []
    for seed in seeds:

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        data_size = 1000

        confounder = GaussianIWAE(feature_size=dim, latent_size=dim, class_size=0, hidden_size=10, num_samples=10).to(device)
        target = SimpleGaussianNN(feature_size=dim, class_size=2 * dim).to(device)
        encoder = SimpleGaussianNN(feature_size=dim, class_size=2 * dim).to(device)

        vb = VariationalBackdoorWithSampling(confounder_model=confounder, 
                                    target_model=target, 
                                    encoder_model=encoder, 
                                    backdoor_samples=10, 
                                    component_samples=10)

        vb.load_state_dict(torch.load(f'trained_models/linear_gaussian/lg_dim={dim}_seed={seed}_no_joint.pt'))

        vb.eval()

        dataset = LinearGaussian(data_size, dim, suppress_print=True, param_seed=seed, generation_seed=99)


        truth = dataset.ground_truth_do_likelihood(dataset.Y, dataset.X)
        truth = np.sum(np.log(truth), axis=-1)

        Y = torch.tensor(dataset.Y, dtype=torch.float32).to(device)
        X = torch.tensor(dataset.X, dtype=torch.float32).to(device)

        variational_est = vb.get_log_backdoor(Y, X, 100, 5).detach().cpu().numpy()
        sampling_est = vb.get_log_backdoor_sampling(Y, X, 100).detach().cpu().numpy()
        t.append(truth.sum())
        v.append(variational_est.sum())
        s.append(sampling_est.sum())
        v_mae.append(mean_absolute_error(variational_est, truth))
        s_mae.append(mean_absolute_error(sampling_est, truth))

    t = np.array(t)
    v = np.array(v)
    s = np.array(s)
    v_mae = np.array(v_mae)
    s_mae = np.array(s_mae)
    
    true_likelihood.append(t.mean())

    lower, upper = confidence_interval(v)
    variational_likelihood['low'].append(lower)
    variational_likelihood['high'].append(upper)
    variational_likelihood['mean'].append(v.mean())

    lower, upper = confidence_interval(s)
    sampling_likelihood['low'].append(lower)
    sampling_likelihood['high'].append(upper)
    sampling_likelihood['mean'].append(s.mean())

    lower, upper = confidence_interval(v_mae)
    variational_err['low'].append(lower)
    variational_err['high'].append(upper)
    variational_err['mean'].append(v_mae.mean())

    lower, upper = confidence_interval(s_mae)
    sampling_err['low'].append(lower)
    sampling_err['high'].append(upper)
    sampling_err['mean'].append(s_mae.mean())

plt.rcParams.update({'font.size': 15})

plt.plot(dims, sampling_likelihood['mean'], label='Sampling', color='orangered')
plt.fill_between(dims, sampling_likelihood['low'], sampling_likelihood['high'], alpha=.5, color='orangered', linewidth=0)
plt.plot(dims, variational_likelihood['mean'], label='Variational Inference', color='limegreen')
plt.fill_between(dims, variational_likelihood['low'], variational_likelihood['high'], alpha=.5, color='limegreen', linewidth=0)
plt.plot(dims, true_likelihood, label='Ground Truth', color='purple')
plt.ylabel('Log Likelihood')
plt.xlabel('Dimension')
plt.legend()
plt.show()

plt.figure()
plt.plot(dims, sampling_likelihood['mean'], label='Sampling', color='orangered')
plt.fill_between(dims, sampling_likelihood['low'], sampling_likelihood['high'], alpha=.5, color='orangered', linewidth=0)
plt.plot(dims, variational_likelihood['mean'], label='Variational Inference', color='limegreen')
plt.fill_between(dims, variational_likelihood['low'], variational_likelihood['high'], alpha=.5, color='limegreen', linewidth=0)
plt.plot(dims, true_likelihood, label='Ground Truth', color='purple')
plt.ylabel('Log Likelihood')
plt.yscale('symlog')
plt.xlabel('Dimension')
plt.legend()
plt.show()

plt.figure()
plt.plot(dims, sampling_err['mean'], label='Sampling', color='orangered')
plt.fill_between(dims, sampling_err['low'], sampling_err['high'], alpha=.5, color='orangered', linewidth=0)
plt.plot(dims, variational_err['mean'], label='Variational Inference', color='limegreen')
plt.fill_between(dims, variational_err['low'], variational_err['high'], alpha=.5, color='limegreen', linewidth=0)
plt.ylabel('Mean Absolute Log Error')
plt.xlabel('Dimension')
plt.legend()
plt.show()


plt.figure()
plt.plot(dims, sampling_err['mean'], label='Sampling', color='orangered')
plt.fill_between(dims, sampling_err['low'], sampling_err['high'], alpha=.5, color='orangered', linewidth=0)
plt.plot(dims, variational_err['mean'], label='Variational Inference', color='limegreen')
plt.fill_between(dims, variational_err['low'], variational_err['high'], alpha=.5, color='limegreen', linewidth=0)
plt.ylabel('Mean Absolute Log Error')
plt.yscale('log')
plt.xlabel('Dimension')
plt.legend()
plt.show()



