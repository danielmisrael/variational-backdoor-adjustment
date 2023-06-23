
from models.backdoor import VariationalBackdoor, SeparateTrainingWrapper, FinetuningWrapper
from models.vae import GaussianIWAE
from models.nn import SimpleGaussianNN 
from data.linear_gaussian import LinearGaussian
import torch
import lightning.pytorch as pl
from lightning.pytorch.loggers import CSVLogger

for seed in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
    for dim in [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]:


        print(f'Dimension, Seed: ({dim}, {seed})')

        torch.manual_seed(seed)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        data_size = 500000

        confounder = GaussianIWAE(feature_size=dim, latent_size=dim, class_size=0, hidden_size=10, num_samples=5).to(device)
        target = SimpleGaussianNN(feature_size=dim, class_size=2 * dim).to(device)
        encoder = SimpleGaussianNN(feature_size=dim, class_size=2 * dim).to(device)

        vb = VariationalBackdoor(confounder_model=confounder, 
                                    target_model=target, 
                                    encoder_model=encoder, 
                                    backdoor_samples=5, 
                                    component_samples=5)


        dataset = LinearGaussian(data_size, dim, suppress_print=True, param_seed=seed)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)

        print('Separate Component Training')
        trainer1 = pl.Trainer(max_epochs=5, default_root_dir='trained_models/')
        trainer1.fit(model=SeparateTrainingWrapper(vb), train_dataloaders=train_loader)
        torch.save(vb.state_dict(), f'trained_models/linear_gaussian/lg_dim={dim}_seed={seed}_no_joint.pt')

        print('Encoder Finetuning')
        vb.load_state_dict(torch.load(f'trained_models/linear_gaussian/lg_dim={dim}_seed={seed}_no_joint.pt'))

        logger2 = CSVLogger('trained_models/linear_gaussian/logs', name='finetuning')
        trainer2 = pl.Trainer(max_epochs=2, default_root_dir='trained_models/', logger=logger2)
        trainer2.fit(model=FinetuningWrapper(vb), train_dataloaders=train_loader)

        torch.save(vb.state_dict(), f'trained_models/linear_gaussian/lg_dim={dim}_seed={seed}_finetuned.pt')




