import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

class Tester:
    def __init__(
        self, 
        model:nn.Module, 
        train_loader:DataLoader, 
        test_loader:DataLoader,
        clf:LogisticRegression,
        device:torch.device
        ):

        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.clf = clf
        self.train_latents = []
        self.train_labels = []
        self.test_latensts = []
        self.test_labels = []

    def extract(self):
        self.model.eval()
        with torch.no_grad():
            for inputs, labels, _ in self.train_loader:
                inputs = inputs.to(self.device, non_blocking=True)
                _, latents = self.model(inputs)
                self.train_latents.append(latents.cpu().detach().numpy())
                self.train_labels.append(labels.cpu().detach().numpy().ravel())
            
            for inputs, labels, _ in self.test_loader:
                inputs = inputs.to(self.device, non_blocking=True)
                _, latents = self.model(inputs)
                self.test_latensts.append(latents.cpu().detach().numpy())
                self.test_labels.append(labels.cpu().detach().numpy().ravel())

        self.train_latents = np.concatenate(self.train_latents)
        self.train_labels = np.concatenate(self.train_labels)
        self.test_latensts = np.concatenate(self.test_latensts)
        self.test_labels = np.concatenate(self.test_labels)

    def fit_clf(self):
        self.clf.fit(self.train_latents, self.train_labels)

    def test(self):
        self.extract()
        self.fit_clf()
        pred = self.clf.predict(self.test_latensts)
        prob = self.clf.predict_proba(self.test_latensts)[:, 1]
        roc = roc_auc_score(self.test_labels, prob)
        rep = classification_report(self.test_labels, pred, target_names=["SR", "AF"],output_dict=True)

        print(f"ROC AUC: {roc:.4f}")
        print("Classification Report:")
        print(classification_report(self.test_labels, pred, target_names=["SR", "AF"]))
       