import copy
import gc
import numpy as np
import os
import torch

from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.config import configure_device, get_lr_scheduler, get_optimizer


# TODO: Convert this to a generic trainer with a step() method instead of train_one_epoch()
class UnsupervisedTrainer:
    def __init__(
        self,
        train_dataset,
        model,
        train_loss,
        val_dataset=None,
        lr_scheduler="poly",
        batch_size=32,
        lr=0.01,
        eval_loss=None,
        log_step=10,
        optimizer="SGD",
        backend="gpu",
        random_state=0,
        optimizer_kwargs={},
        lr_scheduler_kwargs={},
        train_loader_kwargs={},
        val_loader_kwargs={},
        **kwargs,
    ):
        # Create the dataset
        self.lr = lr
        self.random_state = random_state
        self.device = configure_device(backend)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.log_step = log_step
        self.loss_profile = []
        self.batch_size = batch_size
        self.train_loader_kwargs = train_loader_kwargs
        self.val_loader_kwargs = val_loader_kwargs

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            **self.train_loader_kwargs,
        )
        if self.val_dataset is not None:
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                drop_last=False,
                **self.val_loader_kwargs,
            )
        self.model = model.to(self.device)

        # The parameter train_loss must be a callable
        self.train_criterion = train_loss

        # The parameter eval_loss must be a callable
        self.val_criterion = eval_loss

        self.optimizer = get_optimizer(
            optimizer, self.model, self.lr, **optimizer_kwargs
        )
        self.sched_type = lr_scheduler
        self.sched_kwargs = lr_scheduler_kwargs

        # Some initialization code
        torch.manual_seed(self.random_state)
        torch.set_default_tensor_type("torch.FloatTensor")
        if self.device == "gpu":
            # Set a deterministic CuDNN backend
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def train(self, num_epochs, save_path, restore_path=None):
        self.lr_scheduler = get_lr_scheduler(
            self.optimizer, num_epochs, sched_type=self.sched_type, **self.sched_kwargs
        )
        start_epoch = 0
        if restore_path is not None:
            # Load the model
            self.load(restore_path)

        best_eval = 0.0
        tk0 = tqdm(range(start_epoch, num_epochs))
        for epoch_idx in tk0:
            avg_epoch_loss = self.train_one_epoch()

            # LR scheduler step
            self.lr_scheduler.step()

            # Build loss profile
            self.loss_profile.append(avg_epoch_loss)

            # Evaluate the model
            if self.val_criterion is not None:
                val_eval = self.eval()
                tk0.set_postfix_str(
                    f"Avg Loss for epoch: {avg_epoch_loss} Eval Loss: {val_eval}"
                )
                if epoch_idx == 0:
                    best_eval = val_eval
                    self.save(save_path, epoch_idx, prefix="best")
                else:
                    if best_eval > val_eval:
                        # Save this model checkpoint
                        self.save(save_path, epoch_idx, prefix="best")
                        best_eval = val_eval
            else:
                tk0.set_postfix_str(f"Avg Loss for epoch:{avg_epoch_loss}")
                if epoch_idx % 10 == 0:
                    # Save the model every 10 epochs anyways
                    self.save(save_path, epoch_idx)

    def eval(self):
        raise NotImplementedError()

    def train_one_epoch(self):
        raise NotImplementedError()

    def save(self, path, epoch_id, prefix=""):
        checkpoint_name = f"chkpt_{epoch_id}"
        path = os.path.join(path, prefix)
        checkpoint_path = os.path.join(path, f"{checkpoint_name}.pt")
        state_dict = {}
        model_state = copy.deepcopy(self.model.state_dict())
        model_state = {
            k: v.cpu() if isinstance(v, torch.Tensor) else v
            for k, v in model_state.items()
        }
        optim_state = copy.deepcopy(self.optimizer.state_dict())
        for state in optim_state["state"].values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cpu()

        state_dict["model"] = model_state
        state_dict["optimizer"] = optim_state
        state_dict["scheduler"] = self.lr_scheduler.state_dict()
        state_dict["epoch"] = epoch_id + 1
        state_dict["loss_profile"] = self.loss_profile

        os.makedirs(path, exist_ok=True)
        for f in os.listdir(path):
            if f.endswith(".pt"):
                os.remove(os.path.join(path, f))
        torch.save(state_dict, checkpoint_path)
        del model_state, optim_state
        gc.collect()

    def load(self, load_path):
        state_dict = torch.load(load_path)
        iter_val = state_dict.get("epoch", 0)
        self.loss_profile = state_dict.get("loss_profile", [])
        if "model" in state_dict:
            print("Restoring Model state")
            self.model.load_state_dict(state_dict["model"])

        if "optimizer" in state_dict:
            print("Restoring Optimizer state")
            self.optimizer.load_state_dict(state_dict["optimizer"])
            # manually move the optimizer state vectors to device
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)

        if "scheduler" in state_dict:
            print("Restoring Learning Rate scheduler state")
            self.lr_scheduler.load_state_dict(state_dict["scheduler"])

    def update_dataset(self, dataset):
        self.train_dataset = dataset
        # Update the training loader with the new dataset
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            **self.train_loader_kwargs,
        )


class VAETrainer(UnsupervisedTrainer):
    def train_one_epoch(self):
        self.model.train()
        epoch_loss = 0
        tk0 = self.train_loader
        for idx, data_batch in enumerate(tk0):
            self.optimizer.zero_grad()
            data_batch = data_batch.to(self.device)
            _, predictions, mu, logvar = self.model(data_batch)
            loss = self.train_criterion(data_batch, predictions, mu, logvar)
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
        return epoch_loss / len(self.train_loader)

    def eval(self):
        self.model.eval()
        eval_loss = 0
        with torch.no_grad():
            for idx, data_batch in enumerate(self.val_loader):
                self.optimizer.zero_grad()
                data_batch = data_batch.to(self.device)
                _, predictions, mu, logvar = self.model(data_batch)
                loss = self.val_criterion(data_batch, predictions, mu, logvar)
                eval_loss += loss.item()
        return eval_loss / len(self.val_loader)


class AETrainer(UnsupervisedTrainer):
    def train_one_epoch(self):
        self.model.train()
        epoch_loss = 0
        tk0 = self.train_loader
        for idx, data_batch in enumerate(tk0):
            self.optimizer.zero_grad()
            data_batch = data_batch.to(self.device)
            predictions = self.model(data_batch)
            loss = self.train_criterion(data_batch, predictions)
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
        return epoch_loss / len(self.train_loader)


class SparseAETrainer(UnsupervisedTrainer):
    def train_one_epoch(self):
        self.model.train()
        epoch_loss = 0
        tk0 = self.train_loader
        for idx, data_batch in enumerate(tk0):
            self.optimizer.zero_grad()
            data_batch = data_batch.to(self.device)
            z, predictions = self.model(data_batch)
            loss = self.train_criterion(data_batch, predictions) + torch.norm(z, p=1)
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
        return epoch_loss / len(self.train_loader)


class AEMixupTrainer(UnsupervisedTrainer):
    def train_one_epoch(self):
        self.model.train()
        epoch_loss = 0
        tk0 = self.train_loader
        for idx, data_batch in enumerate(tk0):
            self.optimizer.zero_grad()
            data_batch = data_batch.to(self.device)
            if np.random.uniform() > 0.5:
                # Apply mixup in the latent space
                permuted_inds = torch.randperm(self.batch_size).to(self.device)
                permuted_batch = data_batch[permuted_inds]
                lamb = np.random.beta(1.0, 1.0)
                mixup_batch = lamb * data_batch + (1 - lamb) * permuted_batch
                z1 = self.model.encode(data_batch)
                z2 = self.model.encode(permuted_batch)
                z = lamb * z1 + (1 - lamb) * z2
                z_hat = self.model.encode(mixup_batch)
                z_loss = torch.nn.functional.mse_loss(z, z_hat, reduction="mean")
                predictions = self.model(data_batch)
                loss = self.train_criterion(data_batch, predictions) + z_loss
            else:
                predictions = self.model(data_batch)
                loss = self.train_criterion(data_batch, predictions)
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
        return epoch_loss / len(self.train_loader)


class MetricTrainer(UnsupervisedTrainer):
    def train_one_epoch(self):
        self.model.train()
        epoch_loss = 0
        tk0 = self.train_loader
        for idx, (anchor, pos, neg) in enumerate(tk0):
            self.optimizer.zero_grad()
            anchor = anchor.to(self.device)
            pos = pos.to(self.device)
            neg = neg.to(self.device)
            X_anchor = self.model(anchor.float())
            X_pos = self.model(pos.float())
            X_neg = self.model(neg.float())
            loss = self.train_criterion(X_anchor, X_pos, X_neg)
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
        return epoch_loss / len(self.train_loader)
