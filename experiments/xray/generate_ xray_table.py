
from models.backdoor import VariationalBackdoorWithStats
import torch
import numpy as np
from models.nn import ImageGaussianNN
from models.ffjord import FFJORD
from data.xray import XrayCausalDataset
import pickle


def bits_per_dim(log_likelihood, dim):

    logpx_per_dim = log_likelihood / dim
    bits_per_dim = -(logpx_per_dim - np.log(256)) / np.log(2)

    return bits_per_dim


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dim = 28 * 28

def print_model_stats(model_name):

    d = {}

    d['ELBO mean'] = []
    d['E(P(Z)) mean'] =  []
    d['E(P(Y | X, Z)) mean'] = []
    d['E(Q(Z | X, Y)) mean'] = []

    for i in range(5):
        
        confounder = FFJORD(data='xray').to(device)
        target = ImageGaussianNN(feature_size=dim, hidden_size=300, class_size=2 * dim).to(device)
        encoder = ImageGaussianNN(feature_size=dim, hidden_size=300, class_size=2 * dim).to(device)

        vb = VariationalBackdoorWithStats(confounder_model=confounder, 
                                    target_model=target, 
                                    encoder_model=encoder, 
                                    backdoor_samples=1, 
                                    component_samples=1)


        vb.load_state_dict(torch.load(model_name))

        vb.eval()

        data = XrayCausalDataset(split='test')

        b = 1
        k = 1

        batch_size = 100
        loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False)

        vb_elbo = 0
        p_z = 0
        p_y_xz = 0
        q_z_xy = 0
        
        for i, (X, Y, Z) in enumerate(loader):
            if i % 100 == 0:
                print('Progress', i,'/', len(loader))
            with torch.no_grad():
                vb_elbo_batch, p_z_batch, p_y_xz_batch, q_z_xy_batch = vb.backdoor_stats(Y.to(device), X.to(device), b, k)
                vb_elbo += vb_elbo_batch.sum()
                p_z += p_z_batch.sum()
                p_y_xz += p_y_xz_batch.sum()
                q_z_xy += q_z_xy_batch.sum()

        vb_elbo /= len(data)
        p_z /= len(data)
        p_y_xz /= len(data)
        q_z_xy /= len(data)
        
        print('ELBO', vb_elbo.mean())
        print('E(P(Z)) Likelihood', p_z.mean())
        print('E(P(Y | X, Z)) Likelihood', p_y_xz.mean())
        print('E(Q(Z | X, Y)) Likelihood', q_z_xy.mean())


        d['ELBO mean'].append(vb_elbo.mean())
        d['E(P(Z)) mean'].append(p_z.mean())
        d['E(P(Y | X, Z)) mean'].append(p_y_xz.mean())
        d['E(Q(Z | X, Y)) mean'].append(q_z_xy.mean())

    print(d)
    return d

print('Separate Training')
no_joint = 'trained_models/xray/xray_tumor_vb_no_joint.pt'
d1 = print_model_stats(no_joint)

print()

print('Finetuned')
finetuned = 'trained_models/xray/xray_tumor_vb_finetuned.pt'
d2 = print_model_stats(finetuned)

print('Separate')
print(d1)
print('Finetuned')
print(d2)

with open('results/xray/sep_table_results.pickle', 'wb') as handle:
    pickle.dump(d1, handle, protocol=pickle.HIGHEST_PROTOCOL)


with open('results/xray/finetuned_table_results.pickle', 'wb') as handle:
    pickle.dump(d2, handle, protocol=pickle.HIGHEST_PROTOCOL)
