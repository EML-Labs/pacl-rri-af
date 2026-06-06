from typing import Any
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from dotenv import load_dotenv
from torch.utils.data import DataLoader
import torch


@hydra.main(version_base=None, config_path="configs", config_name="conf")
def main(cfg : DictConfig) -> None:
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate model, load checkpoint
    model = hydra.utils.instantiate(cfg.model).to(device)
    state_dict = torch.load(to_absolute_path(cfg.checkpoint.path), map_location=device, weights_only=True)
    model.load_state_dict(state_dict)

    # Instantiate datasets, dataloaders, classifier, and tester
    train_dataset = hydra.utils.instantiate(cfg.dataset, train=True,test=False)
    test_dataset = hydra.utils.instantiate(cfg.dataset, test=True,train=False)
    train_loader = hydra.utils.instantiate(cfg.dataloder, dataset=train_dataset,shuffle=True)
    test_loader = hydra.utils.instantiate(cfg.dataloder, dataset=test_dataset,shuffle=False)
    clf = hydra.utils.instantiate(cfg.clf)
    tester = hydra.utils.instantiate(cfg.tester, model=model, train_loader=train_loader, test_loader=test_loader, clf=clf, device=device)

    # Run testing
    tester.test()

if __name__ == "__main__":
    load_dotenv()
    main()