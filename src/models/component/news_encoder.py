import copy

import torch
import torch.nn as nn
import numpy as np
from models.base.layers import *
from torch_geometric.nn import Sequential, GCNConv
from pathlib import Path

import copy

import torch
import torch.nn as nn
import numpy as np
from models.base.layers import *
import pandas as pd
import ast

class NewsEncoder(nn.Module):
    def __init__(self, cfg, glove_emb=None):
        super().__init__()
        token_emb_dim = cfg.model.word_emb_dim
        self.news_dim = cfg.model.head_num * cfg.model.head_dim

        if cfg.dataset.dataset_lang == 'english':
            pretrain = torch.from_numpy(glove_emb).float()
            self.word_encoder = nn.Embedding.from_pretrained(pretrain, freeze=False, padding_idx=0)
        elif cfg.dataset.dataset_lang == 'danish':
            df = pd.read_parquet('data/bert_base_multilingual_cased.parquet')
            embeddings = df['google-bert/bert-base-multilingual-cased'].apply(lambda x: list(x)).tolist()
            pretrain = torch.tensor(embeddings).float()  # Ensure it's a 2D tensor
            self.word_encoder = nn.Embedding.from_pretrained(pretrain, freeze=False, padding_idx=0)
        else:
            self.word_encoder = nn.Embedding(glove_emb+1, 300, padding_idx=0)
            nn.init.uniform_(self.word_encoder.weight, -1.0, 1.0)

        self.view_size = [cfg.model.title_size, cfg.model.abstract_size]
        

        self.attention = Sequential('x, mask', [
            (nn.Dropout(p=cfg.dropout_probability), 'x -> x'),
            (MultiHeadAttention(token_emb_dim,
                                token_emb_dim,
                                token_emb_dim,
                                cfg.model.head_num,
                                cfg.model.head_dim), 'x,x,x,mask -> x'),
            nn.LayerNorm(self.news_dim),
            nn.Dropout(p=cfg.dropout_probability),

            (AttentionPooling(self.news_dim,
                                cfg.model.attention_hidden_dim), 'x,mask -> x'),
            nn.LayerNorm(self.news_dim),
            # nn.Linear(self.news_dim, self.news_dim),
            # nn.LeakyReLU(0.2),
        ])


    def forward(self, news_input, mask=None):
        batch_size = news_input.shape[0]
        num_news = news_input.shape[1]

        title_input, _, _, _, _ = news_input.split([self.view_size[0], 5, 1, 1, 1], dim=-1)
        title_word_emb = self.word_encoder(title_input.long().view(-1, self.view_size[0]))

        total_word_emb = title_word_emb

        result = self.attention(total_word_emb, mask)

        return result.view(batch_size, num_news, self.news_dim)