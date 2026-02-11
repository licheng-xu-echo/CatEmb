import torch,logging,math
from torch import nn
from torch.optim.lr_scheduler import _LRScheduler
from torch.nn import MSELoss
from .loss import dual_CL
mse_loss = MSELoss()
def grad_norm(m):
    return math.sqrt(sum([p.grad.norm().item() ** 2 for p in m.parameters() if p.grad is not None]))

def get_lr(optimizer):
    current_lr = optimizer.param_groups[0]["lr"]
    return current_lr

class NoamLR(_LRScheduler):
    """
    Adapted from https://github.com/tugstugi/pytorch-saltnet/blob/master/utils/lr_scheduler.py

    Implements the Noam Learning rate schedule. This corresponds to increasing the learning rate
    linearly for the first ``warmup_steps`` training steps, and decreasing it thereafter proportionally
    to the inverse square root of the step number, scaled by the inverse square root of the
    dimensionality of the model. Time will tell if this is just madness or it's actually important.
    Parameters
    ----------
    warmup_steps: ``int``, required.
        The number of steps to linearly increase the learning rate.
    """
    def __init__(self, optimizer, model_size, warmup_steps):
        self.model_size = model_size
        self.warmup_steps = warmup_steps
        super().__init__(optimizer)

    def get_lr(self):
        step = max(1, self._step_count)
        scale = self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup_steps**(-1.5))

        return [base_lr * scale for base_lr in self.base_lrs]

class CLTrain():
    def __init__(self,clmodel, param_3d, param_cl, param_optimizer, param_scheduler, param_other):
        
        self.clmodel = clmodel
        self.clmodel.to(param_other['device'])
        self.metric = param_cl['metric']
        self.T = param_cl['T']
        self.cl_weight = param_cl['cl_weight']
        self.kl_weight = param_cl['kl_weight']
        self.e_weight = param_cl['e_weight']
        self.param_3d = param_3d
        self.param_optimizer = param_optimizer
        self.param_scheduler = param_scheduler
        self.param_other = param_other
        self.init_optimizer()
        self.init_scheduler()

    def init_optimizer(self):
        if self.param_optimizer['type'].lower() == 'adamw':
            optimizer = torch.optim.AdamW(
                self.clmodel.parameters(),
                lr=self.param_optimizer['lr'], amsgrad=True,
                weight_decay=1e-12)
        elif self.param_optimizer['type'].lower() == 'adam':
            optimizer = torch.optim.Adam(
                self.clmodel.parameters(),
                lr=self.param_optimizer['lr'], amsgrad=True,
                weight_decay=1e-12)
        else:
            raise NotImplementedError(f"Optimizer {self.param_optimizer['type']} not implemented")
        self.optimizer = optimizer
        return optimizer
    
    def init_scheduler(self):
        if self.param_scheduler['type'].lower() == 'steplr':
            scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                        step_size=self.param_scheduler['lr_decay_step_size'],
                                                        gamma=self.param_scheduler['lr_decay_factor'])
        elif self.param_scheduler['type'].lower() == 'noamlr':
            if self.param_3d['model_type'] == 'equif':
                model_size = self.param_3d['sphere_channels']
            elif self.param_3d['model_type'] == 'dimenetpp':
                model_size = self.param_3d['out_channels']
            else:
                raise NotImplementedError(f"Model {self.param_3d['model_type']} not implemented")

            scheduler = NoamLR(self.optimizer,model_size=model_size,
                                    warmup_steps=self.param_scheduler['warmup_step'])
        elif self.param_scheduler['type'].lower() == 'reduceonplateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                   mode='min',
                                                                   factor=self.param_scheduler['lr_decay_factor'],
                                                                   patience=self.param_scheduler['lr_decay_step_size'],
                                                                   verbose=False,
                                                                   min_lr=self.param_scheduler['min_lr'])
        else:
            raise NotImplementedError(f"Scheduler {self.param_scheduler['type']} not implemented")
        self.scheduler = scheduler
        return scheduler
    
    def train(self,dataloader):
        self.clmodel.train()
        loss_accum = 0.
        cl_loss_accum = 0.
        kl_loss_accum = 0.
        energy_loss_accum = 0.
        mol_num = 0
        for step, data in enumerate(dataloader):
            data = data.to(self.param_other['device'])
            self.optimizer.zero_grad()
            mol_feat_from_2d, mol_feat_from_3d, energy_p = self.clmodel(data)
            cl_loss, kl_loss = dual_CL(mol_feat_from_2d, mol_feat_from_3d, normalize=True, metric=self.metric, T=self.T,
                                          lambda_cl=self.cl_weight, lambda_kl=self.kl_weight)
            energy_loss = mse_loss(energy_p, data.E)
            loss = cl_loss * self.cl_weight + kl_loss * self.kl_weight + energy_loss * self.e_weight
            loss.backward()
            nn.utils.clip_grad_norm_(self.clmodel.parameters(), self.param_other['clip_norm'])
            self.optimizer.step()
            if self.param_scheduler['type'].lower() == 'noamlr':
                self.scheduler.step()
            g_norm = grad_norm(self.clmodel)
            lr_cur = get_lr(self.optimizer)
            self.clmodel.zero_grad()

            loss_item = loss.detach().cpu().item()
            cl_loss_item = cl_loss.detach().cpu().item()
            kl_loss_item = kl_loss.detach().cpu().item()
            energy_loss_item = energy_loss.detach().cpu().item()
            
            mol_num += len(mol_feat_from_2d)
            loss_accum += loss_item
            cl_loss_accum += cl_loss_item
            kl_loss_accum += kl_loss_item
            energy_loss_accum += energy_loss_item

            if (step+1) % self.param_other['log_iter_step'] == 0:
                logging.info(f"Step [{step+1}/{len(dataloader)}], loss: {loss_item/len(mol_feat_from_2d):.4f}, cl loss: {cl_loss_item/len(mol_feat_from_2d):.4f}, kl loss: {kl_loss_item/len(mol_feat_from_2d):.6f}, E loss: {energy_loss_item/len(mol_feat_from_2d):.4f}, g_norm: {g_norm:.4f}, lr: {lr_cur:.8f}")
        
        loss_ave = loss_accum / mol_num
        cl_loss_ave = cl_loss_accum / mol_num
        kl_loss_ave = kl_loss_accum / mol_num
        energy_loss_ave = energy_loss_accum / mol_num
        
        g_norm = grad_norm(self.clmodel)
        lr_cur = get_lr(self.optimizer)
        logging.info(f"Train epoch [{self.epoch+1}/{self.param_other['epoch']}], loss: {loss_ave:.4f}, cl loss: {cl_loss_ave:.4f}, kl loss: {kl_loss_ave:.6f}, energy loss: {energy_loss_ave:.4f}, g_norm: {g_norm:.4f}, lr: {lr_cur:.8f}")
        if self.param_scheduler['type'].lower() == 'reduceonplateau':
            self.scheduler.step(loss_ave)
        elif self.param_scheduler['type'].lower() == 'steplr':
            self.scheduler.step()
        elif self.param_scheduler['type'].lower() == 'noamlr':
            pass
        else:
            raise NotImplementedError(f"Scheduler {self.param_scheduler['type']} not implemented")
        return loss_ave, cl_loss_ave, kl_loss_ave, energy_loss_ave
    
    def eval(self,dataloader):
        self.clmodel.eval()
        with torch.no_grad():
            loss_accum = 0.
            cl_loss_accum = 0.
            kl_loss_accum = 0.
            energy_loss_accum = 0.
            mol_num = 0
            for step, data in enumerate(dataloader):
                data = data.to(self.param_other['device'])
                mol_feat_from_2d, mol_feat_from_3d, energy_p = self.clmodel(data)
                cl_loss, kl_loss = dual_CL(mol_feat_from_2d, mol_feat_from_3d, normalize=True, metric=self.metric, T=self.T,
                                          lambda_cl=self.cl_weight, lambda_kl=self.kl_weight)
                energy_loss = mse_loss(energy_p, data.E)
                loss = cl_loss * self.cl_weight + kl_loss * self.kl_weight + energy_loss * self.e_weight
                
                loss_item = loss.detach().cpu().item()
                cl_loss_item = cl_loss.detach().cpu().item()
                kl_loss_item = kl_loss.detach().cpu().item()
                energy_loss_item = energy_loss.detach().cpu().item()
                
                mol_num += len(mol_feat_from_2d)
                loss_accum += loss_item
                cl_loss_accum += cl_loss_item
                kl_loss_accum += kl_loss_item
                energy_loss_accum += energy_loss_item
                
            loss_ave = loss_accum / mol_num
            cl_loss_ave = cl_loss_accum / mol_num
            kl_loss_ave = kl_loss_accum / mol_num
            energy_loss_ave = energy_loss_accum / mol_num
            logging.info(f"Eval loss: {loss_ave:.4f} ({self.best_val_loss:.4f}), cl loss: {cl_loss_ave:.4f}, kl loss: {kl_loss_ave:.6f}, energy loss: {energy_loss_ave:.4f}")
        return loss_ave, cl_loss_ave, kl_loss_ave, energy_loss_ave
    def save_model(self):
        ckpt_file = f"{self.param_other['save_path']}/best_model.pt"
        logging.info(f"!!!!!!!!! Saving model to {ckpt_file}... !!!!!!!!!")
        checkpoint = {'epoch': self.epoch,
                      'model_state_dict': self.clmodel.state_dict(),
                      'optimizer_state_dict': self.optimizer.state_dict(),
                      'scheduler_state_dict': self.scheduler.state_dict(),
                      'best_val_loss': self.best_val_loss,}
        torch.save(checkpoint, ckpt_file)

    def run(self, train_dataloader, val_dataloader, test_dataloader=None):
        self.best_val_loss = 99999999999999.
        self.clmodel.zero_grad()
        for epoch in range(self.param_other['epoch']):
            self.epoch = epoch
            logging.info(f"Epoch {epoch+1}/{self.param_other['epoch']}, training...")
            train_loss, train_cl_loss, train_kl_loss, train_energy_loss = self.train(train_dataloader)
            val_loss, val_cl_loss, val_kl_loss, val_energy_loss = self.eval(val_dataloader)
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_model()
        logging.info(f"Training finished, best val loss: {self.best_val_loss:.4f}")
