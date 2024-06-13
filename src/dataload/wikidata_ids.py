
import graphvite as gv
import pickle
import pandas as pd
from tqdm import tqdm


articles = pd.read_parquet("data/ebnerd_small/articles.parquet")

entity_strs = []
entities = []
for idx, line in tqdm(articles.iterrows()):
    title = line['title']
    t_entity_str = set()
    for ent in line['ner_clusters']: # add only entities that are in the title
        for name in title:
            if name in ent:
                t_entity_str.add(ent)
    t_entity_str = list(t_entity_str)
    entity_strs.append(t_entity_str)


with open("data/transe_wikidata5m.pkl", "rb") as fin:
        model = pickle.load(fin)
for t_entity_str_ in entity_strs:
    entity2id = model.graph.entity2id
    alias2entity = gv.dataset.wikidata5m.alias2entity
    entity_ids = [entity2id[alias2entity[ent]] for ent in t_entity_str_]
    entities.append(entity_ids)

articles['ner_ids'] = entities
articles.to_parquet("data/ebnerd_small/articles.parquet_ids.parquet")