import torch
from tqdm import tqdm
from CBraMod.finetune_evaluator import Evaluator
from torch.nn import CrossEntropyLoss
from timeit import default_timer as timer
import numpy as np
import copy
from sklearn.metrics import balanced_accuracy_score



def get_metrics_for_binaryclass(model, data_loader, device="mps"):
    model.eval()

    truths = []
    preds = []
    scores = []
    for x, y in tqdm(data_loader, mininterval=1):
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        score_y = torch.sigmoid(pred)
        pred_y = torch.argmax(score_y, dim=1)
        truths += y.long().cpu().squeeze().numpy().tolist()
        preds += pred_y.cpu().squeeze().numpy().tolist()
        scores += score_y.cpu().numpy().tolist()

    truths = np.array(truths)
    preds = np.array(preds)
    scores = np.array(scores)
    acc = balanced_accuracy_score(truths, preds)
    return acc


class Trainer(object):
    def __init__(self, params, data_loader, model):
        self.params = params
        self.device = params.device
        self.data_loader = data_loader
        self.model = model

        self.val_eval = Evaluator(params, self.data_loader['val'])
        self.test_eval = Evaluator(params, self.data_loader['test'])
        
        self.criterion = CrossEntropyLoss(label_smoothing=self.params.label_smoothing).cuda()
        self.best_model_states = None

        backbone_params = []
        other_params = []
        for name, param in self.model.named_parameters():
            if "backbone" in name:
                backbone_params.append(param)

                if params.frozen:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            else:
                other_params.append(param)

        if self.params.optimizer == 'AdamW':
            if self.params.multi_lr: # set different learning rates for different modules
                self.optimizer = torch.optim.AdamW([
                    {'params': backbone_params, 'lr': self.params.lr},
                    {'params': other_params, 'lr': self.params.lr * 5}
                ], weight_decay=self.params.weight_decay)
            else:
                self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.params.lr,
                                                   weight_decay=self.params.weight_decay)
        else:
            if self.params.multi_lr:
                self.optimizer = torch.optim.SGD([
                    {'params': backbone_params, 'lr': self.params.lr},
                    {'params': other_params, 'lr': self.params.lr * 5}
                ],  momentum=0.9, weight_decay=self.params.weight_decay)
            else:
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.params.lr, momentum=0.9,
                                                 weight_decay=self.params.weight_decay)

        self.data_length = len(self.data_loader['train'])
        self.optimizer_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.params.epochs * self.data_length, eta_min=1e-6
        )

    def train_for_binaryclass(self):
        best_acc = 0
        best_epoch = 0
        for epoch in range(self.params.epochs):
            self.model.train()
            start_time = timer()
            losses = []
            for x, y in tqdm(self.data_loader['train'], mininterval=10):
                self.optimizer.zero_grad()
                x = x.to(self.device)
                y = y.to(self.device).long()
                pred = self.model(x)

                loss = self.criterion(pred, y)

                loss.backward()
                losses.append(loss.data.cpu().numpy())
                if self.params.clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.clip_value)
                    # torch.nn.utils.clip_grad_value_(self.model.parameters(), self.params.clip_value)
                self.optimizer.step()
                self.optimizer_scheduler.step()

            optim_state = self.optimizer.state_dict()

            with torch.no_grad():
                acc = get_metrics_for_binaryclass(self.model, self.data_loader['val'], self.device)
                print(
                    "Epoch {} : Training Loss: {:.5f}, acc: {:.5f}, LR: {:.5f}, Time elapsed {:.2f} mins".format(
                        epoch + 1,
                        np.mean(losses),
                        acc,
                        optim_state['param_groups'][0]['lr'],
                        (timer() - start_time) / 60
                    )
                )
                if acc > best_acc:
                    best_acc = acc
                    best_epoch = epoch + 1
                    self.best_model_states = copy.deepcopy(self.model.state_dict())

        self.model.load_state_dict(self.best_model_states)
        with torch.no_grad():
            print("***************************Test************************")
            acc = get_metrics_for_binaryclass(self.model, self.data_loader['val'], self.device)
            print("***************************Test results************************")
            print("Test Evaluation: acc: {:.5f}".format(acc))