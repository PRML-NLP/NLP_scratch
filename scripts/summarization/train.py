import argparse
import torch, os
import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch import optim
# from pytorch_lightning.core.lightning import LightningModule
from models.pointer_generator import PointerGenerator
from dataset import CnnDailyMailDataset
from config import configs
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR

class TrainingModel(pl.LightningModule):
    def __init__(self, model_name, lr, config):
        super().__init__()
        self.lr = lr
        if model_name == 'pointer-generator':
            self.model = PointerGenerator(**config)
        else:
            raise NotImplementedError(model_name + ' is not available.')
    
    def training_step(self, batch, batch_idx):
        return self.model(batch)       
            
    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            loss = self.model(batch)
            self.log('val_loss', loss.item())
            # metrics = {'val_loss': loss.item(),}
            # self.log_dict(metrics)
                
    def test_step(self, batch, batch_idx):
        self.model.generate(batch)
        raise NotImplementedError
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adagrad(self.parameters(), lr=(self.lr or self.learning_rate), initial_accumulator_value=0.1)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ExponentialLR(optimizer, gamma=0.99),
                "interval": "step"
            }
        }
    

def main(args):
    pl.seed_everything(args.seed)
    
    config = configs[args.model_name]
    
    train_dataset = CnnDailyMailDataset(args.data_path, args.vocab_path, max_enc_len=400, max_dec_len=100, mode='train', cached=True)
    val_dataset = CnnDailyMailDataset(args.data_path, args.vocab_path, max_enc_len=400, max_dec_len=120, mode='val', cached=True)
    # test_dataset = CnnDailyMailDataset(args.data_path, args.vocab_path, max_enc_len=400, max_dec_len=120, mode='test', cached=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=2)
    # test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size)
    
    config['max_oovs'] = train_dataset.max_art_oovs
    config['pad_id'] = train_dataset.vocab.encode(train_dataset.special_tokens['pad'])
    
    model = TrainingModel(args.model_name, args.lr, config)
        
    checkpoint_callback = ModelCheckpoint(
        filename=os.path.join(args.ckpt_save_path, '{epoch:d}'),
        verbose=True, save_last=True, save_top_k=args.save_top_k,
        monitor='val_loss',mode='min'
    )
    # default patience: 3
    early_stopping = EarlyStopping(
        monitor='val_loss', verbose=True, mode='min')
    
    train_config = {
        'callbacks': [checkpoint_callback, early_stopping],
        'max_epochs': args.n_epochs,
        'gpus': args.n_gpus,
        'auto_lr_find': True,
        'gradient_clip_val': 2.0
    }
    if args.resume is not None:
        train_config['resume_from_checkpoint'] = args.resume
        
    trainer = pl.Trainer(**train_config)
    trainer.fit(model, train_loader, val_loader)
    
    
if __name__=='__main__':
    parser = argparse.ArgumentParser('Text summarization')
    parser.add_argument('--model_name', type=str, default='pointer-generator', help='Neural model name')
    parser.add_argument('--data_path', type=str, default='data/cnn_dm')
    parser.add_argument('--vocab_path', type=str, default='data/vocab.pkl')
    parser.add_argument('--seed', type=int, default=777)
    parser.add_argument('--n_epochs', type=int, default=33)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--test_batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.15)
    parser.add_argument('--n_gpus', type=int, default=1, help='The number of gpus')
    # Checkpoints
    parser.add_argument('--ckpt_save_path', type=str, default='checkpoints', help='Path to save checkpoints')
    parser.add_argument('--resume', type=str, default=None, help='Checkpoint file path')
    parser.add_argument('--save_top_k', type=int, default=3, help='The number of top checkpoints to save')
        
    args = parser.parse_args()
    
    if not os.path.exists(args.ckpt_save_path):
        os.mkdir(args.ckpt_save_path)
        
    main(args)