import os
import pathlib
from typing import Union
import numpy as np
import torch
from torch.nn import Module
from torch import optim
from torch_geometric.loader import DataLoader
from lips.dataset import DataSet
from lips.augmented_simulators.torch_simulator import TorchSimulator
from lips.augmented_simulators.torch_models.utils import OPTIMIZERS
from matplotlib import pyplot as plt

class GnnSimulator(TorchSimulator):
    """GNN simulator allowing to train and predict using a GNN model

    Parameters
    ----------
    TorchSimulator : _type_
        _description_
    """
    def __init__(self,
                 model: Module,
                 sim_config_path: Union[pathlib.Path, str],
                 name: Union[str, None]=None,
                 scaler: None = None,
                 log_path: Union[str, None]=None,
                 seed: int=42,
                 **kwargs):
        super().__init__(model, sim_config_path, name, scaler, log_path, seed, **kwargs)
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.test_ood_loader = None
        
    def train(self,
              train_dataset: DataSet,
              val_dataset: Union[None, DataSet]=None,
              save_path: Union[None, str] = None,
              **kwargs):
        self.params.update(kwargs)
        self._model.params.update(kwargs)
        
        if self.train_loader is None:
            self.train_loader = self._model.process_dataset(train_dataset)
        if (val_dataset is not None) and (self.val_loader is None):
            self.val_loader = self._model.process_dataset(val_dataset)
        
        self.build_model()
        self._model.to(self.params["device"])
        
        optimizer = self._get_optimizer(optimizer=OPTIMIZERS[self.params["optimizer"]["name"]],
                                        **self.params["optimizer"]["params"])
        
        # for metric_ in self.params["metrics"]:
        #     self.train_metrics[metric_] = list()
        #     if val_loader is not None:
        #         self.val_metrics[metric_] = list()
                
        self.logger.info("Training of {%s} started", self.name)

        for epoch in range(self.params["epochs"]):
            #pbar.set_description("Epoch %s" % str(epoch))
            train_loss_epoch = self._train_one_epoch(epoch, self.train_loader, optimizer, **kwargs)
            self.train_losses.append(train_loss_epoch)
            # for nm_, arr_ in self.train_metrics.items():
            #     arr_.append(train_metrics_epoch[nm_])

            if self.val_loader is not None:
                val_loss_epoch = self._validate(self.val_loader)
                self.val_losses.append(val_loss_epoch)
                # for nm_, arr_ in self.val_metrics.items():
                #     arr_.append(val_metrics_epoch[nm_])

            # check point
            if self.params["save_freq"] and (save_path is not None):
                if epoch % self.params["ckpt_freq"] == 0:
                    self.save(save_path, epoch)

        self.trained = True
        # save the final model
        if save_path:
            self.save(save_path)
            
    def _train_one_epoch(self, epoch:int, train_loader: DataLoader, optimizer: optim.Optimizer, **kwargs) -> set:
        """
        train the model at a epoch
        """
        self._model.train()
        torch.set_grad_enabled(True)

        total_loss = 0
        # metric_dict = dict()
        # for metric in self.params["metrics"]:
        #     metric_dict[metric] = 0

        #pbar=tqdm(train_loader)
        for batch_ in train_loader:
        #for batch_ in pbar:
            #pbar.set_description("Batch within epoch (Training)")
            optimizer.zero_grad()
            prediction, errors = self._model._do_forward(batch_)
            # loss_func = self._model.get_loss_func(loss_name=self.params["loss"]["name"], **self.params["loss"]["params"])
            # loss = loss_func(prediction, target)
            if self.params["train_with_discount"]:
                gamma = self.params["gamma"]
                K = len(errors)
                coefs = torch.zeros(K)
                coefs = torch.tensor([torch.power(gamma, K-k) for k in range(K)])
                new_errors = [(coef * error) for coef, error in zip(coefs, errors)]
                loss = torch.stack(new_errors).sum() # sum over layers (k)
            else:
                loss = torch.stack(errors).sum()
            loss.backward()
            optimizer.step()
            total_loss += (loss * len(batch_.x))
                                
            # for metric in self.params["metrics"]:
            #     metric_func = self._model.get_loss_func(loss_name=metric, reduction="mean")
            #     metric_value = metric_func(prediction, batch_.y)
            #     metric_value = metric_value.item()*len(batch_.y)
            #     metric_dict[metric] += metric_value

        mean_loss = total_loss.item()/len(train_loader.dataset)
        # for metric in self.params["metrics"]:
        #     metric_dict[metric] /= len(train_loader.dataset)
        # print(f"Train Epoch: {epoch}   Avg_Loss: {mean_loss:.5f}",
        #       [f"{metric}: {metric_dict[metric]:.5f}" for metric in self.params["metrics"]])
        # return mean_loss, metric_dict
        print(f"Train Epoch: {epoch}   Avg_Loss: {mean_loss:.5f}")
        return mean_loss
    
    def _validate(self, val_loader: DataLoader, **kwargs) -> set:
        """function used for validation of the model

        It is separated from evaluate function, because it should be called at each epoch during training

        Parameters
        ----------
        val_loader : DataLoader
            _description_

        Returns
        -------
        set
            _description_

        Raises
        ------
        NotImplementedError
            _description_
        """
        self.params.update(kwargs)
        self._model.eval()
        total_loss = 0
        # metric_dict = dict()
        # for metric in self.params["metrics"]:
        #     metric_dict[metric] = 0

        with torch.no_grad():
            for batch_ in val_loader:
                _, errors = self._model._do_forward(batch_)
                if self.params["train_with_discount"]:
                    gamma = self.params["gamma"]
                    K = len(errors)
                    coefs = torch.zeros(K)
                    coefs = torch.tensor([torch.power(gamma, K-k) for k in range(K)])
                    new_errors = [(coef * error) for coef, error in zip(coefs, errors)]
                    loss = torch.stack(new_errors).sum() # sum over layers (k)
                else:
                    loss = torch.stack(errors).sum()
                total_loss += (loss * len(batch_.x))
                

                # for metric in self.params["metrics"]:
                #     metric_func = self._model.get_loss_func(loss_name=metric, reduction="mean")
                #     metric_value = metric_func(prediction, target)
                #     metric_value = metric_value.item()*len(target)
                #     metric_dict[metric] += metric_value

        mean_loss = total_loss.item()/len(val_loader.dataset)
        # for metric in self.params["metrics"]:
        #     metric_dict[metric] /= len(val_loader.dataset)
        # print(f"Eval:   Avg_Loss: {mean_loss:.5f}",
        #       [f"{metric}: {metric_dict[metric]:.5f}" for metric in self.params["metrics"]])
        print(f"Eval:   Avg_Loss: {mean_loss:.5f}")
        return mean_loss#, metric_dict
    
    def predict(self, dataset: DataSet, reconstruct_output: bool=True, **kwargs):
        """
        predictions of GNN based model
        """
        if "eval_batch_size" in kwargs:
            self.params["eval_batch_size"] = kwargs["eval_batch_size"]
            self._model.params["eval_batch_size"] = kwargs["eval_batch_size"]
        
        if dataset.name == "val":
            if self.val_loader is None:
                self.val_loader = self._model.process_dataset(dataset, training=False, **kwargs)
                test_loader = self.val_loader
            else:
                test_loader = self.val_loader
                
        if dataset.name == "test": 
            if self.test_loader is None:
                self.test_loader = self._model.process_dataset(dataset, training=False, **kwargs)
                test_loader = self.test_loader
            else:
                test_loader = self.test_loader

        if dataset.name == "test_ood_topo":  
            if self.test_ood_loader is None:
                self.test_ood_loader = self._model.process_dataset(dataset, training=False, **kwargs)
                test_loader = self.test_ood_loader
            else:
                test_loader = self.test_ood_loader
            
        
        self._model.eval() 
        predictions = []
        observations = []
        total_loss = 0
        
        with torch.no_grad():
            #pbar=tqdm(test_loader)
            #for batch_ in pbar:
            for batch_ in test_loader:
                #pbar.set_description("Batch (Prediction)")
                prediction, errors = self._model._do_forward(batch_)
                
                predictions.append(prediction)
                observations.append(batch_.y)

                # try:
                #     predictions.append(prediction.numpy())
                #     observations.append(batch_.y.numpy())
                # except TypeError:
                #     predictions.append(prediction.cpu().data.numpy())
                #     observations.append(batch_.y.cpu().data.numpy())

                # loss = loss_func(prediction, target)
                # total_loss += loss.item()*len(target)

                # for metric in self.params["metrics"]:
                #     metric_func = self._model.get_loss_func(loss_name=metric, reduction="mean")
                #     metric_value = metric_func(prediction, target)
                #     metric_value = metric_value.item()*len(target)
                #     metric_dict[metric] += metric_value

        # mean_loss = total_loss/len(test_loader.dataset)
        # for metric in self.params["metrics"]:
        #     metric_dict[metric] /= len(test_loader.dataset)
        #print(f"Eval:   Avg_Loss: {mean_loss:.5f}",
        #      [f"{metric}: {metric_dict[metric]:.5f}" for metric in self.params["metrics"]])

        # predictions = np.concatenate(predictions)
        # observations = np.concatenate(observations)
        predictions = torch.vstack(predictions)
        observations = torch.vstack(observations)
        
        if reconstruct_output:
            predictions = self._model._post_process(predictions)
            predictions = self._model._reconstruct_output(predictions, dataset.data)
            observations = self._model._reconstruct_output(observations, dataset.data)

        self._predictions[dataset.name] = predictions
        self._observations[dataset.name] = observations
        
        return predictions
    
    def _get_optimizer(self, optimizer: optim.Optimizer=optim.Adam, **kwargs):
        return optimizer(self._model.parameters(), **kwargs)
    
    def save(self):
        pass
    
    def visualize_convergence(self, figsize=(15,5), save_path: str=None):
        """Visualizing the convergence of the model
        """
        # raise an error if the train_losses is empty
        if len(self.train_losses) == 0:
            raise RuntimeError("The model should be trained before visualizing the convergence")
        num_metrics = len(self.params["metrics"])
        if num_metrics == 0:
            nb_subplots = 1
        else:
            nb_subplots = num_metrics + 1
        fig, ax = plt.subplots(1,nb_subplots, figsize=figsize)
        ax[0].set_title("MSE")
        ax[0].plot(self.train_losses, label='train_loss')
        if len(self.val_losses) > 0:
            ax[0].plot(self.val_losses, label='val_loss')
        for idx_, metric_name in enumerate(self.params["metrics"]):
            ax[idx_+1].set_title(metric_name)
            ax[idx_+1].plot(self.train_metrics[metric_name], label=f"train_{metric_name}")
            if len(self.val_metrics[metric_name]) > 0:
                ax[idx_+1].plot(self.val_metrics[metric_name], label=f"val_{metric_name}")
        for i in range(nb_subplots):
            ax[i].grid()
            ax[i].legend()
        # save the figure
        if save_path is not None:
            fig.savefig(save_path)
            