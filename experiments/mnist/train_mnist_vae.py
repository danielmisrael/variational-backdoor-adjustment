
from models.vae import BernoulliIWAE, ConditionalIWAETrainingWrapper
import torch
import lightning.pytorch as pl
from data.mnist import BinaryMNISTCausalDataset

dim = 28 * 28
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = BernoulliIWAE(feature_size=dim, latent_size=50, class_size=dim, hidden_size=400, num_samples=10).to(device)
dataset = BinaryMNISTCausalDataset()
train_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)


print('Train Conditional VAE')
trainer1 = pl.Trainer(max_epochs=30, default_root_dir='trained_models/')
trainer1.fit(model=ConditionalIWAETrainingWrapper(model), train_dataloaders=train_loader)

torch.save(model.state_dict(), f'trained_models/mnist/mnist_vae.pt')