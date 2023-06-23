from models.backdoor import VariationalBackdoor, SeparateTrainingWrapper ,FinetuningWrapper
from models.vae import BernoulliIWAE
from models.nn import BernoulliNN
import torch
import lightning.pytorch as pl
from data.mnist import BinaryMNISTCausalDataset

dim = 28 * 28
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

confounder = BernoulliIWAE(feature_size=dim, latent_size=50, class_size=0, hidden_size=400, num_samples=10).to(device)
target = BernoulliIWAE(feature_size=dim, latent_size=50, class_size=2*dim, hidden_size=400, num_samples=10).to(device)
encoder = BernoulliNN(feature_size=dim, hidden_size=400, class_size=2 * dim).to(device)

vb = VariationalBackdoor(confounder_model=confounder, 
                            target_model=target, 
                            encoder_model=encoder, 
                            backdoor_samples=10, 
                            component_samples=10)




dataset = BinaryMNISTCausalDataset()
train_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)


print('Separate Component Training')
trainer1 = pl.Trainer(max_epochs=50, default_root_dir='trained_models/')
trainer1.fit(model=SeparateTrainingWrapper(vb), train_dataloaders=train_loader)

torch.save(vb.state_dict(), f'trained_models/mnist/mnist_vb_no_joint.pt')

print('Encoder Finetuning')
vb.load_state_dict(torch.load(f'trained_models/mnist/mnist_vb_no_joint.pt'))
trainer2 = pl.Trainer(max_epochs=5, default_root_dir='trained_models/')
trainer2.fit(model=FinetuningWrapper(vb), train_dataloaders=train_loader)

torch.save(vb.state_dict(), f'trained_models/mnist/mnist_vb_finetuned.pt')















# Old stuff ffjord
# import torch
# from ffjord.load_mnist import load_fford
# from vbdata.mnist_toy import MNISTCausalDataset
# import matplotlib.pyplot as plt

# import math

# def standard_normal_logprob(z):
#     logZ = -0.5 * math.log(2 * math.pi)
#     return logZ - z.pow(2) / 2

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model, train_loader = load_fford()
# model = model.to(device)
# model.eval()




# dataset = MNISTCausalDataset()

# train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
# cvt = lambda x: x.type(torch.float32).to(device, non_blocking=True)

# for (X, Y, Z) in train_loader:
#     Z = cvt(Z).to(device)


#     # print('max', torch.max(Z))
#     # print(Z)

#     zero = torch.zeros(Z.size(0), 1).to(device)

#     # print(zero.size())

#     z, delta_logp = model(Z, zero)

#     logpz = standard_normal_logprob(z).view(z.size(0), -1).sum(1, keepdim=True)

#     log_likelihood = logpz - delta_logp

#     print('pz', logpz)

#     print('px', log_likelihood)





