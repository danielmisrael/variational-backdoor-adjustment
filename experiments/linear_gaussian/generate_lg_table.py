
from models.backdoor import VariationalBackdoorWithStats
from models.vae import GaussianIWAE
from models.nn import SimpleGaussianNN
from data.linear_gaussian import LinearGaussian
import torch
import numpy as np
import pickle


def print_stats(lg, vb, b, k):

    X_data = torch.tensor(lg.X, dtype=torch.float32).to(device)
    Y_data = torch.tensor(lg.Y, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        vb_elbo, p_z, p_y_xz, q_z_xy = vb.backdoor_stats(Y_data, X_data, b, k)
    
    X = lg.X
    Y = lg.Y
    Z = lg.Z

    Z_data = torch.tensor(Z, dtype=torch.float32).to(device)
    

    # Backdoor Error
    backdoor_truth = np.log(lg.ground_truth_do_likelihood(Y, X)).sum(axis=1)
    error = np.abs(vb_elbo - backdoor_truth)
    print('Backdoor Error', error.mean())
    print('E(P(Z)) Likelihood', p_z.mean())
    print('E(P(Y | X, Z)) Likelihood', p_y_xz.mean())
    print('E(Q(Z | X, Y)) Likelihood', q_z_xy.mean())

    p_z = vb.confounder.get_log_likelihood(Z_data, None, k=k)
    print('P(Z)', p_z.mean().item())

    p_y_g_xz = vb.target.get_log_likelihood(Y_data, torch.cat([X_data, Z_data], 1))
    print('P(Y | X, Z)', p_y_g_xz.mean().item())

    q_z_g_xy = vb.encoder.get_log_likelihood(Z_data, torch.cat([X_data, Y_data], 1))
    print('Q(Z | X, Y)', q_z_g_xy.mean().item())

    d = {}

    d['ELBO mean'] = vb_elbo.mean().item()
    d['Backdoor error mean'] = error.mean().item()
    d['E(P(Z)) mean'] =  p_z.mean().item()
    d['E(P(Y | X, Z)) mean'] = p_y_xz.mean().item()
    d['E(Q(Z | X, Y)) mean'] = q_z_xy.mean().item()

    d['ELBO stderr'] = vb_elbo.std().item() / np.sqrt(data_size)
    d['Backdoor error stderr'] = error.std().item() / np.sqrt(data_size)
    d['E(P(Z)) stderr'] =  p_z.std().item() / np.sqrt(data_size)
    d['E(P(Y | X, Z)) stderr'] = p_y_xz.std().item() / np.sqrt(data_size)
    d['E(Q(Z | X, Y)) stderr'] = q_z_xy.std().item() / np.sqrt(data_size)

    return d

    

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dim = 15
seed = 1

data_size = 10000

def print_model_stats(model_name, save_name):

    for i in range(10):

        confounder = GaussianIWAE(feature_size=dim, latent_size=dim, class_size=0, hidden_size=10, num_samples=10).to(device)
        target = SimpleGaussianNN(feature_size=dim, class_size=2 * dim).to(device)
        encoder = SimpleGaussianNN(feature_size=dim, class_size=2 * dim).to(device)

        vb = VariationalBackdoorWithStats(confounder_model=confounder, 
                                    target_model=target, 
                                    encoder_model=encoder, 
                                    backdoor_samples=10, 
                                    component_samples=10)

        vb.load_state_dict(torch.load(model_name))

        vb.eval()

        # In Distribution
        print('In Distribution')
        lg = LinearGaussian(data_size, dim, suppress_print=True, param_seed=seed, generation_seed=99)

        np.random.seed(99 + i)
        d = print_stats(lg, vb, 100, 5)

        with open(f'results/lg/{save_name}_id_seed={i}.pickle', 'wb') as handle:
            pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Out of Distribution
        print('Out of Distribution')
        bound_z = 7
        np.random.seed(99)
        do_z = np.random.uniform(-bound_z, bound_z, (data_size, dim))
        lg.generate_linear_gaussian_backdoor(data_size, dim,  
            do_z=do_z, seed=99)
        
        np.random.seed(99 + i)
        d= print_stats(lg, vb, 100, 5)

        with open(f'results/lg/{save_name}_od_seed={i}.pickle', 'wb') as handle:
            pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)


print('Separate Training')
no_joint = f'trained_models/linear_gaussian/lg_dim={dim}_seed={seed}_no_joint.pt'
print_model_stats(no_joint, 'sep')

print()

print('Finetuned')
finetuned = f'trained_models/linear_gaussian/lg_dim={dim}_seed={seed}_finetuned.pt'
print_model_stats(finetuned, 'finetuned')
