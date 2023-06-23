from models.backdoor import VariationalBackdoor, SeparateTrainingWrapper ,FinetuningWrapper
from models.nn import ImageGaussianNN
from models.ffjord import FFJORD
import torch
import lightning.pytorch as pl
from data.xray import XrayCausalDataset
from lightning.pytorch.loggers import CSVLogger

dim = 28 * 28
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

confounder = FFJORD(data='xray').to(device)
target = ImageGaussianNN(feature_size=dim, hidden_size=200, class_size=2 * dim).to(device)
encoder = ImageGaussianNN(feature_size=dim, hidden_size=200, class_size=2 * dim).to(device)

vb = VariationalBackdoor(confounder_model=confounder, 
                            target_model=target, 
                            encoder_model=encoder, 
                            backdoor_samples=1, 
                            component_samples=1)




dataset = XrayCausalDataset()
train_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)


print('Separate Component Training')
logger1 = CSVLogger('trained_models/xray/logs', name='separate_training')
trainer1 = pl.Trainer(max_epochs=20, default_root_dir='trained_models/', logger=logger1)
trainer1.fit(model=SeparateTrainingWrapper(vb, ignore_confounder=True), train_dataloaders=train_loader)

torch.save(vb.state_dict(), f'trained_models/xray/xray_vb_no_joint.pt')

print('Encoder Finetuning')
vb.load_state_dict(torch.load(f'trained_models/xray/vb_tumor_xray_no_joint.pt'))
logger2 = CSVLogger("trained_models/xray/logs", name="finetuning")
trainer2 = pl.Trainer(max_epochs=2, default_root_dir='trained_models/', logger=logger2)
trainer2.fit(model=FinetuningWrapper(vb, lr=1e-4), train_dataloaders=train_loader)

torch.save(vb.state_dict(), f'trained_models/xray/xray_vb_finetuned.pt')