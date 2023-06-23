import numpy as np
import torch
from data.mnist import BinaryMNISTCausalDataset
from models.nn import BernoulliNN
from models.vae import BernoulliIWAE
from models.backdoor import VariationalBackdoor
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import colors



def construct_comparison_matrix(model_fn):
    matrix = np.zeros((10, 10))

    for x in range(10):
        for z in range(10):

            dataset = BinaryMNISTCausalDataset(do_x=x, do_z=z, test=True)
            batch_size = 10
            num_samples = 100
            test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

            likelihood = 0
            for i, (X, Y, Z) in enumerate(test_loader):
                if i < num_samples:
                    with torch.no_grad():
                        likelihood += model_fn(Y.to(device), X.to(device)).sum()

            likelihood = likelihood / (num_samples * batch_size)
            print('(z, x)', (z, x))
            print('likelihood', likelihood)
            matrix[x][z] = likelihood

    return matrix

def plot_matrix(data):

    divnorm=colors.TwoSlopeNorm(vcenter=0, vmin=-40, vmax=40)

    sns.heatmap(data, linewidth=0.5, cmap='seismic', norm=divnorm)
    plt.xlabel('do(X)')
    plt.ylabel('Z')
    plt.show()


vae_matrix = np.load('misc/vae_matrix.npy')
print(vae_matrix)

dim = 28 * 28
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

vae = BernoulliIWAE(feature_size=dim, latent_size=50, class_size=dim, hidden_size=400, num_samples=10).to(device)
vae.load_state_dict(torch.load('trained_models/mnist/mnist_vae.pt'))
vae.eval()
vae_fn = lambda y, x: vae.get_log_likelihood(y, x, k=10)

matrix = construct_comparison_matrix(vae_fn)

with open('misc/vae_matrix.npy', 'wb') as f:
    np.save(f, matrix)



confounder = BernoulliIWAE(feature_size=dim, latent_size=50, class_size=0, hidden_size=400, num_samples=10).to(device)
target = BernoulliIWAE(feature_size=dim, latent_size=50, class_size=2*dim, hidden_size=400, num_samples=10).to(device)
encoder = BernoulliNN(feature_size=dim, hidden_size=400, class_size=2 * dim).to(device)

vb = VariationalBackdoor(confounder_model=confounder, 
                            target_model=target, 
                            encoder_model=encoder, 
                            backdoor_samples=10, 
                            component_samples=5)


vb.load_state_dict(torch.load('trained_models/mnist/mnist_vb_finetuned.pt'))
vb.eval()
vb_fn = lambda y, x: vb.get_log_backdoor(y, x, 10, 10)

matrix = construct_comparison_matrix(vb_fn)

with open('misc/vb_matrix_finetuned.npy', 'wb') as f:
    np.save(f, matrix)



vb_matrix = np.load('misc/vb_matrix_finetuned.npy')
vae_matrix = np.load('misc/vae_matrix.npy')

plt.rcParams.update({'font.size': 23})


plt.figure()
ax = sns.heatmap(vb_matrix, linewidth=0.5, cmap='viridis')
cbar = ax.collections[0].colorbar
plt.xlabel('do(X)')
plt.ylabel('Z')
plt.tight_layout()
plt.show()

plt.figure()
ax = sns.heatmap(vae_matrix, linewidth=0.5, cmap='viridis')
cbar = ax.collections[0].colorbar
plt.xlabel('do(X)')
plt.ylabel('Z')
plt.tight_layout()
plt.show()


plt.figure()
divnorm=colors.TwoSlopeNorm(vcenter=0, vmin=-40, vmax=40)
ax = sns.heatmap((vb_matrix - vae_matrix), linewidth=0.5, cmap='seismic', norm=divnorm)
cbar = ax.collections[0].colorbar
cbar.set_label('Log Likelihood', rotation=270, labelpad=40)
plt.xlabel('do(X)')
plt.ylabel('Z')
plt.tight_layout()
plt.show()



