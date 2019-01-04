from __future__ import print_function

import torch

import time
import copy

from core.types import EpochPhase
from core.utils import device


class ModelTrainer:
    """
      The ModelTrainer class handles training and validation of a given model.
    """

    def __init__(
            self,
            model,
            dataloaders,
            criterion,
            optimizer,
            number_epoches=25,
            is_inception=False
    ):
        """
        :param model:
               A PyTorch Model
        :param dataloaders:
               A dictionary of dataloaders
        :param criterion:
               A loss function
        :param optimizer:
               An optimizer
        :param number_epoches:
               Number of epoches to run
        :param is_inception:
               Whether accommodating the 'Inception V3' model, which uses an auxiliary
               output the overall model loss respects both the auxiliary and final output.
        """

        self.model = model
        self.dataloaders = dataloaders
        self.criterion = criterion
        self.optimizer = optimizer
        self.number_epoches = number_epoches
        self.is_inception = is_inception
        self.since = time.time()
        self.val_acc_history = []
        self.best_model_wts = copy.deepcopy(model.state_dict())
        self.best_acc = 0.0

    def _calc_loss_and_corrects(
            self,
            phase: EpochPhase,
            inputs,
            labels,
            enable_grad: bool,
            has_auxiliary_outputs: bool
    ):
        inputs = inputs.to(device)
        labels = labels.to(device)
        # zero the parameter gradients
        self.optimizer.zero_grad()
        with torch.set_grad_enabled(enable_grad):
            # Get model outputs and calculate loss
            # Special case for inception because in training it has an
            # auxiliary output.
            # In train mode we calculate the loss by summing the final
            # output and the auxiliary output
            # but in testing we only consider final output
            if has_auxiliary_outputs:
                outputs, aux_outputs = self.model(inputs)
                regular_loss = self.criterion(outputs, labels)
                aux_loss = self.criterion(aux_outputs, labels)
                loss = regular_loss + 0.4 * aux_loss
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)

            # backward + optimize only if in training phase
            if phase is EpochPhase.Training:
                loss.backward()
                self.optimizer.step()

        _loss = (loss.item() * inputs.size(0))
        _corrects = torch.sum(preds == labels.data)
        return _loss, _corrects

    def _update_model_for_phase(self, phase: EpochPhase):
        if phase is EpochPhase.Training:
            self.model.train()  # Set model to training mode
        elif phase is EpochPhase.Evaluating:
            self.model.eval()  # Set model to evaluating mode
        else:
            raise TypeError(f"Not supported phase {phase}")

        running_loss = 0.0
        running_corrects = 0
        data_for_this_phase = self.dataloaders[phase]
        data_len = len(data_for_this_phase.dataset)

        has_auxiliary_outputs = self.is_inception and phase is EpochPhase.Training
        enable_grad = phase == EpochPhase.Training
        for inputs, labels in data_for_this_phase:
            phase_loss, phase_corrects = self._calc_loss_and_corrects(
                phase=phase,
                enable_grad=enable_grad,
                has_auxiliary_outputs=has_auxiliary_outputs,
                inputs=inputs,
                labels=labels
            )
            # statistics
            running_loss = running_loss + phase_loss
            running_corrects = running_corrects + phase_corrects

        epoch_loss = running_loss / data_len
        epoch_acc = running_corrects.double() / data_len

        print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} ')

        if phase is EpochPhase.Evaluating:
            self.val_acc_history.append(epoch_acc)
            if epoch_acc > self.best_acc:
                self.best_acc = epoch_acc
                self.best_model_wts = copy.deepcopy(self.model.state_dict())

    def _run_one_epoch(self, epoch):
        print(f'Epoch {epoch}/{self.number_epoches - 1}')
        print('-' * 10)
        for phase in [EpochPhase.Training, EpochPhase.Evaluating]:
            self._update_model_for_phase(phase)

    def train(self):
        """
         Trains for specified number of epoches.
         After each epoch, runs a full validation step.
        :return:
           The best performing model and validation accuracy history
        """
        for epoch in range(self.number_epoches):
            self._run_one_epoch(epoch)
            print()
        time_elapsed = time.time() - self.since
        print(f'Training completed in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {self.best_acc:.4f}')

        # load best model weights
        self.model.load_state_dict(self.best_model_wts)
        return self.model, self.val_acc_history
