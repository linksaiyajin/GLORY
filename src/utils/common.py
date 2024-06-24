"""
Common utils and tools.
"""
import pickle
import random

import pandas as pd
import torch
import numpy as np
import pyrootutils
from pathlib import Path
import torch.distributed as dist
from tqdm import tqdm

import importlib
from omegaconf import DictConfig, ListConfig

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)

def load_model(cfg):
    framework = getattr(importlib.import_module(f"models.{cfg.model.model_name}"), cfg.model.model_name)

    if cfg.model.use_entity:
        entity_dict = pickle.load(open(Path(cfg.dataset.train_dir) / "entity_dict.bin", "rb"))
        entity_embedding_dict = pickle.load(open(Path(cfg.dataset.train_dir) / "entity_embedding_dict.bin", "rb"))
        entity_emb = load_pretrain_emb(entity_dict, entity_embedding_dict, 300)
    else:
        entity_emb = None

    if cfg.dataset.dataset_lang == 'english':
        word_dict = pickle.load(open(Path(cfg.dataset.train_dir) / "word_dict.bin", "rb"))
        glove_emb_matrix = load_pretrain_emb(word_dict, cfg.model.word_emb_dim)
        glove_emb = torch.from_numpy(glove_emb_matrix).float()
    else:
        word_dict = pickle.load(open(Path(cfg.dataset.train_dir) / "word_dict.bin", "rb"))
        glove_emb = len(word_dict)
    
    model = framework(cfg, glove_emb=glove_emb, entity_emb=entity_emb)

    return model


def save_model(cfg, model, optimizer=None, mark=None):
    file_path = Path(f"{cfg.path.ckp_dir}/{cfg.model.model_name}_{cfg.dataset.dataset_name}_{mark}.pth")
    file_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
        },
        file_path)
    print(f"Model Saved. Path = {file_path}")


def load_pretrain_emb(target_dict, target_embedding_dict, target_dim):
    print("WORKING!!!!!!!")
    embedding_matrix = np.zeros(shape=(len(target_dict) + 1, target_dim))

    for ent_id, index in tqdm(target_dict.items(), desc="Processing embeddings"):
        emb = target_embedding_dict[ent_id]
        embedding_np = emb.cpu().numpy()
        embedding_matrix[index] = embedding_np
    
    print('-----------------------------------------------------')
    print(f'Dict length: {len(target_dict)}')
    return embedding_matrix


def reduce_mean(result, nprocs):
    rt = result.detach()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt


def pretty_print(d, indent=0):
    for key, value in d.items():
        print('\t' * indent + str(key)+ '\t' + str(value))


def get_root():
    return pyrootutils.setup_root(
        search_from=__file__,
        indicator=[".git", "README.md"],
        pythonpath=True,
        dotenv=True,
    )


class EarlyStopping:
    """
    Early Stopping class
    """

    def __init__(self, patience=3):
        self.patience = patience
        self.counter = 0
        self.best_score = 0.0

    def __call__(self, score):
        """
        The greater score, the better result. Be careful the symbol.
        """
        if score > self.best_score:
            early_stop = False
            get_better = True
            self.counter = 0
            self.best_score = score
        else:
            get_better = False
            self.counter += 1
            if self.counter >= self.patience:
                early_stop = True
            else:
                early_stop = False

        return early_stop, get_better
