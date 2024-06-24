import collections
import os
from pathlib import Path
from nltk.tokenize import word_tokenize
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

import torch.nn.functional as F
from tqdm import tqdm
from pandas import read_parquet, to_numeric
import random
import pickle
from collections import Counter
import numpy as np
import torch
import json
import itertools

from flair.embeddings import WordEmbeddings
from flair.data import Sentence

import spacy


# Initialize Danish embeddings using Flair
danish_embedding = WordEmbeddings('da')

nlp = spacy.load('da_core_news_md')

def update_dict(target_dict, key, value=None):
    if key not in target_dict:
        if value is None:
            target_dict[key] = len(target_dict) + 1
        else:
            target_dict[key] = value


def get_sample(all_elements, num_sample):
    if num_sample > len(all_elements):
        return random.sample(all_elements * (num_sample // len(all_elements) + 1), num_sample)
    else:
        return random.sample(all_elements, num_sample)


def prepare_distributed_data(cfg, mode="train"):
    data_dir = {"train": cfg.dataset.train_dir, "val": cfg.dataset.val_dir, "test": cfg.dataset.test_dir}
    target_file = os.path.join(data_dir[mode], f"behaviors_np{cfg.npratio}_0.tsv")
    if os.path.exists(target_file) and not cfg.reprocess:
        return 0
    print(f'Target_file is not exist. New behavior file in {target_file}')

    behaviors = []
    behavior_file_path = os.path.join(data_dir[mode], 'behaviors.parquet')
    history_file_path = os.path.join(data_dir[mode], 'history.parquet')

    if mode == 'train':
        f = read_parquet(behavior_file_path)
        hist = read_parquet(history_file_path)
        for _, line in tqdm(f.iterrows()):
            iid = line['impression_id']
            uid = line['user_id']
            time = line['impression_time']
            history = hist[hist['user_id'] == uid]['article_id_fixed'].values[0]
            imp = list(line['article_ids_inview'])
            click = line['article_ids_clicked']
            click_one_hot = [0] * len(imp)
            click_id = [imp.index(c) for c in click]
            for c in click_id:
                click_one_hot[c] = 1
            impressions = list(zip(imp, click_one_hot))
            history = ' '.join(str(h) for h in history)
            pos, neg = [], []

            for news_ID, label in impressions:
                if label == '0':
                    neg.append(news_ID)
                elif label == '1':
                    pos.append(news_ID)
            if len(pos) == 0 or len(neg) == 0:
                continue
            for pos_id in pos:
                neg_candidate = get_sample(neg, cfg.npratio)
                neg_str = ' '.join(neg_candidate)
                new_line = '\t'.join([iid, uid, time, history, pos_id, neg_str]) + '\n'
                behaviors.append(new_line)
        random.shuffle(behaviors)

        behaviors_per_file = [[] for _ in range(cfg.gpu_num)]
        for i, line in enumerate(behaviors):
            behaviors_per_file[i % cfg.gpu_num].append(line)

    elif mode in ['val', 'test']:
        behaviors_per_file = [[] for _ in range(cfg.gpu_num)]
        f = read_parquet(behavior_file_path)
        hist = read_parquet(history_file_path)
        lines = []
        for _, line in tqdm(f.iterrows()):
            iid = line['impression_id']
            uid = line['user_id']
            time = line['impression_time']
            history = hist[hist['user_id'] == uid]['article_id_fixed'].values[0]
            imp = list(line['article_ids_inview'])
            click = line['article_ids_clicked']
            click_one_hot = [0] * len(imp)
            click_id = [imp.index(c) for c in click]
            for c in click_id:
                click_one_hot[c] = 1
            impressions = list(zip(imp, click_one_hot))
            new_line = '\t'.join([str(iid), str(uid), str(time), ' '.join(str(h) for h in history), ' '.join('-'.join([str(id), str(i)]) for id, i in impressions)]) + '\n'
            lines.append(new_line)

        for i, line in enumerate(tqdm(lines)):
            behaviors_per_file[i % cfg.gpu_num].append(line)

    print(f'[{mode}]Writing files...')
    for i in range(cfg.gpu_num):
        processed_file_path = os.path.join(data_dir[mode], f'behaviors_np{cfg.npratio}_{i}.tsv')
        with open(processed_file_path, 'w') as f:
            f.writelines(behaviors_per_file[i])

    return len(behaviors)


def read_raw_news(cfg, file_path, mode='train'):
    """
    Function for reading the raw news file, news.tsv

    Args:
        cfg:
        file_path(Path):                path of news.tsv
        mode(string, optional):        train or test


    Returns:
        tuple:     (news, news_index, category_dict, subcategory_dict, word_dict)

    """
    import nltk
    nltk.download('punkt')
    
    data_dir = {"train": cfg.dataset.train_dir, "val": cfg.dataset.val_dir, "test": cfg.dataset.test_dir}

    if mode in ['val', 'test']:
        news_dict = pickle.load(open(Path(data_dir["train"]) / "news_dict.bin", "rb"))
        entity_dict = pickle.load(open(Path(data_dir["train"]) / "entity_dict.bin", "rb"))
        news = pickle.load(open(Path(data_dir["train"]) / "nltk_news.bin", "rb"))
    else:
        news = {}
        news_dict = {}
        entity_dict = {}

    category_dict = {}
    subcategory_dict = {}
    word_cnt = Counter()  # Counter is a subclass of the dictionary dict.

    articles = read_parquet("data/ebnerd_small/articles_danish_ner.parquet")

    
    for idx, line in tqdm(articles.iterrows(), desc=f"[{mode}]Processing raw news"):
        # print("lines", line)
        news_id = line['article_id']
        category = line['category_str']
        subcategory = line['subcategory'].flat[0] if len(line['subcategory']) != 0 else None# if (isinstance(line['subcategory'], np.ndarray) and len(line['subcategory']) != 0) else line['subcategory'] # TODO: only has ID for now, not the name
        # subcategory = None
        title = line['title']
        # print('title', title)
        docs = nlp(title)

        ents = docs.ents
        # print("ents", ents)

        abstract = line['subtitle']
        url = line['url']

        t_entity_str = line['ner_ids']

        update_dict(target_dict=news_dict, key=news_id)

        entity_ids = []

        # Entity
        if ents:
            # [update_dict(target_dict=entity_dict, key=entity_id) for entity_id in entity_ids]
            # print("WORKING!!!!", ents)
            # print("t_entity_str", t_entity_str)
            ide = 0
            for entity in ents:
                sentence = Sentence(entity.text)
                danish_embedding.embed(sentence)
                idt = 0
                for token in sentence:
                    entity_id = str(idx) + "_" + str(ide) + "_" + str(idt)  # Ensure unique hashable ID
                    entity_dict[entity_id] = token.embedding
                    # print("entity_id", entity_id)
                    entity_ids.append(entity_id)
                    idt += 1
                ide += 1
        else:
            entity_ids = []
        
        tokens = word_tokenize(title.lower(), language=cfg.dataset.dataset_lang)

        update_dict(target_dict=news, key=news_id, value=[tokens, category, subcategory, entity_ids,
                                                            news_dict[news_id]])

        if mode == 'train':
            update_dict(target_dict=category_dict, key=category)
            update_dict(target_dict=subcategory_dict, key=subcategory)
            word_cnt.update(tokens)

    if mode == 'train':
        word = [k for k, v in word_cnt.items() if v > cfg.model.word_filter_num]
        word_dict = {k: v for k, v in zip(word, range(1, len(word) + 1))}
        return news, news_dict, category_dict, subcategory_dict, entity_dict, word_dict
    else:  # val, test
        return news, news_dict, None, None, entity_dict, None


def read_parsed_news(cfg, news, news_dict, category_dict=None, subcategory_dict=None, entity_dict=None, word_dict=None):
    # print("entity_dict", entity_dict)
    news_num = len(news) + 1
    print("news_num", news_num) 
    news_category, news_subcategory, news_index = [np.zeros((news_num, 1), dtype='int32') for _ in range(3)]
    news_entity = np.zeros((news_num, 5), dtype='int32')

    news_title = np.zeros((news_num, cfg.model.title_size), dtype='int32')

    for _news_id in tqdm(news, total=len(news), desc="Processing parsed news"):
        _title, _category, _subcategory, _entity_ids, _news_index = news[_news_id]
        # print("news[_news_id]", news[_news_id]) 
        # print(_title, _category, _subcategory, _entity_ids, _news_index)

        news_category[_news_index, 0] = category_dict[_category] if _category in category_dict else 0
        news_subcategory[_news_index, 0] = subcategory_dict[_subcategory] if _subcategory in subcategory_dict else 0
        news_index[_news_index, 0] = news_dict[_news_id]

        # if _entity_ids:
            # entity_embeddings = []
            # ids = []
            # for entity_id in _entity_ids:
            #     print('entity_id', entity_id)
            #     if entity_id in entity_dict:
            #         entity_embeddings.append(entity_dict[entity_id])
            #         print('entity_id', entity_dict[entity_id])
            # if entity_embeddings.size > 0:
            # entity_index = [entity_dict[entity_id] if entity_id in entity_dict else 0 for entity_id in _entity_ids]
            # print("entity_index", entity_index)
            # print("_entity_ids[:cfg.model.entity_size]", _entity_ids[:cfg.model.entity_size])
        news_entity[_news_index, :min(cfg.model.entity_size, len(_entity_ids))] = _entity_ids[:cfg.model.entity_size]

        for _word_id in range(min(cfg.model.title_size, len(_title))):
            if _title[_word_id] in word_dict:
                news_title[_news_index, _word_id] = word_dict[_title[_word_id]]

    print(news_entity)
    return news_title, news_entity, news_category, news_subcategory, news_index


def prepare_preprocess_bin(cfg, mode):
    data_dir = {"train": cfg.dataset.train_dir, "val": cfg.dataset.val_dir, "test": cfg.dataset.test_dir}

    if cfg.reprocess is True:
        nltk_news, nltk_news_dict, category_dict, subcategory_dict, entity_dict, word_dict = read_raw_news(
            file_path=Path(data_dir[mode]) / "articles.parquet",
            cfg=cfg,
            mode=mode,
        )

        if mode == "train":
            pickle.dump(category_dict, open(Path(data_dir[mode]) / "category_dict.bin", "wb"))
            pickle.dump(subcategory_dict, open(Path(data_dir[mode]) / "subcategory_dict.bin", "wb"))
            pickle.dump(word_dict, open(Path(data_dir[mode]) / "word_dict.bin", "wb"))
        else:
            category_dict = pickle.load(open(Path(data_dir["train"]) / "category_dict.bin", "rb"))
            subcategory_dict = pickle.load(open(Path(data_dir["train"]) / "subcategory_dict.bin", "rb"))
            word_dict = pickle.load(open(Path(data_dir["train"]) / "word_dict.bin", "rb"))

        pickle.dump(entity_dict, open(Path(data_dir[mode]) / "entity_dict.bin", "wb"))
        # print(entity_dict)
        pickle.dump(nltk_news, open(Path(data_dir[mode]) / "nltk_news.bin", "wb"))
        pickle.dump(nltk_news_dict, open(Path(data_dir[mode]) / "news_dict.bin", "wb"))
        nltk_news_features = read_parsed_news(cfg, nltk_news, nltk_news_dict, category_dict, subcategory_dict, entity_dict, word_dict)
        news_input = np.concatenate([x for x in nltk_news_features], axis=1)
        pickle.dump(news_input, open(Path(data_dir[mode]) / "nltk_token_news.bin", "wb"))
        print("Glove token preprocess finish.")
    else:
        print(f'[{mode}] All preprocessed files exist.')


def prepare_news_graph(cfg, mode='train'):
    data_dir = {"train": cfg.dataset.train_dir, "val": cfg.dataset.val_dir, "test": cfg.dataset.test_dir}

    nltk_target_path = Path(data_dir[mode]) / "nltk_news_graph.pt"

    reprocess_flag = False
    if nltk_target_path.exists() is False:
        reprocess_flag = True
        
    if (reprocess_flag == False) and (cfg.reprocess == False):
        print(f"[{mode}] All graphs exist !")
        return
    
    # -----------------------------------------News Graph------------------------------------------------
    # behavior_path = Path(data_dir['train']) / "behaviors.tsv"
    behavior_path = os.path.join(data_dir['train'], 'behaviors.parquet')
    origin_graph_path = Path(data_dir['train']) / "nltk_news_graph.pt"

    news_dict = pickle.load(open(Path(data_dir[mode]) / "news_dict.bin", "rb"))
    nltk_token_news = pickle.load(open(Path(data_dir[mode]) / "nltk_token_news.bin", "rb"))
    
    # ------------------- Build Graph -------------------------------
    if mode == 'train':
        edge_list, user_set = [], set()
        # num_line = len(open(behavior_path, encoding='utf-8').readlines())
        # with open(behavior_path, 'r', encoding='utf-8') as f:
        history_file_path = os.path.join(data_dir[mode], 'history.parquet')
        f = read_parquet(behavior_path)
        hist = read_parquet(history_file_path)
        lines = []
        for _,line in tqdm(f.iterrows()):
            iid = line['impression_id']
            uid = line['user_id']
            time = line['impression_time']
            history = hist[hist['user_id'] == uid]['article_id_fixed'].values[0]
            imp = list(line['article_ids_inview'])
            click = line['article_ids_clicked']
            click_one_hot = [0] * len(imp)
            click_id = [imp.index(c) for c in click]
            for c in click_id:
                click_one_hot[c] = 1
            impressions = list(zip(imp, click_one_hot))
            new_line = '\t'.join([str(iid), str(uid), str(time), ' '.join(str(h) for h in history), ' '.join('-'.join([str(id), str(i)]) for id,i in impressions)]) + '\n'
            lines.append(new_line)

        # for line in tqdm(f, total=num_line, desc=f"[{mode}] Processing behaviors news to News Graph"):
        for line in tqdm(lines, desc=f"[{mode}] Processing behaviors news to News Graph"):
            # print("lines", lines)
            line = line.strip().split('\t')

            # check duplicate user
            used_id = line[1]
            if used_id in user_set:
                continue
            else:
                user_set.add(used_id)

            # record cnt & read path
            history = line[3].split()
            if len(history) > 1:
                # print("history1", history)
                long_edge = [news_dict[int(news_id)] for news_id in history]
                edge_list.append(long_edge)

            # print("history2", history)

        # edge count
        node_feat = nltk_token_news
        target_path = nltk_target_path
        num_nodes = len(news_dict) + 1

        short_edges = []
        for edge in tqdm(edge_list, total=len(edge_list), desc=f"Processing news edge list"):
            # Trajectory Graph
            if cfg.model.use_graph_type == 0:
                for i in range(len(edge) - 1):
                    short_edges.append((edge[i], edge[i + 1]))
                    # short_edges.append((edge[i + 1], edge[i]))
            elif cfg.model.use_graph_type == 1:
                # Co-occurence Graph
                for i in range(len(edge) - 1):
                    for j in range(i+1, len(edge)):
                        short_edges.append((edge[i], edge[j]))
                        short_edges.append((edge[j], edge[i]))
            else:
                assert False, "Wrong"

        edge_weights = Counter(short_edges)
        unique_edges = list(edge_weights.keys())

        edge_index = torch.tensor(list(zip(*unique_edges)), dtype=torch.long)
        edge_attr = torch.tensor([edge_weights[edge] for edge in unique_edges], dtype=torch.long)

        data = Data(x=torch.from_numpy(node_feat),
                edge_index=edge_index, edge_attr=edge_attr,
                num_nodes=num_nodes)
    
        torch.save(data, target_path)
        print(data)
        print(f"[{mode}] Finish News Graph Construction, \nGraph Path: {target_path} \nGraph Info: {data}")
    
    elif mode in ['test', 'val']:
        origin_graph = torch.load(origin_graph_path)
        edge_index = origin_graph.edge_index
        edge_attr = origin_graph.edge_attr
        node_feat = nltk_token_news

        data = Data(x=torch.from_numpy(node_feat),
                    edge_index=edge_index, edge_attr=edge_attr,
                    num_nodes=len(news_dict) + 1)
        
        torch.save(data, nltk_target_path)
        print(f"[{mode}] Finish nltk News Graph Construction, \nGraph Path: {nltk_target_path}\nGraph Info: {data}")


def prepare_neighbor_list(cfg, mode='train', target='news'):
    print(f"[{mode}] Start to process neighbors list")

    data_dir = {"train": cfg.dataset.train_dir, "val": cfg.dataset.val_dir, "test": cfg.dataset.test_dir}

    neighbor_dict_path = Path(data_dir[mode]) / f"{target}_neighbor_dict.bin"
    weights_dict_path = Path(data_dir[mode]) / f"{target}_weights_dict.bin"

    reprocess_flag = False
    for file_path in [neighbor_dict_path, weights_dict_path]:
        if file_path.exists() is False:
            reprocess_flag = True

    if (reprocess_flag == False) and (cfg.reprocess == False) and (cfg.reprocess_neighbors == False):
        print(f"[{mode}] All {target} Neighbor dict exist !")
        return

    if target == 'news':
        target_graph_path = Path(data_dir[mode]) / "nltk_news_graph.pt"
        target_dict = pickle.load(open(Path(data_dir[mode]) / "news_dict.bin", "rb"))
        graph_data = torch.load(target_graph_path)
    elif target == 'entity':
        target_graph_path = Path(data_dir[mode]) / "entity_graph.pt"
        target_dict = pickle.load(open(Path(data_dir[mode]) / "entity_dict.bin", "rb"))
        graph_data = torch.load(target_graph_path)
    else:
        assert False, f"[{mode}] Wrong target {target} "

    edge_index = graph_data.edge_index
    print('edge_index', edge_index)
    edge_attr = graph_data.edge_attr
    print('edge_attr', edge_attr)

    if cfg.model.directed is False:
        edge_index, edge_attr = to_undirected(edge_index, edge_attr)

    neighbor_dict = collections.defaultdict(list)
    neighbor_weights_dict = collections.defaultdict(list)

    for i in range(1, len(target_dict) + 1):
        dst_edges = torch.where(edge_index[1] == i)[0]
        neighbor_weights = edge_attr[dst_edges]
        neighbor_nodes = edge_index[0][dst_edges]
        sorted_weights, indices = torch.sort(neighbor_weights, descending=True)
        neighbor_dict[i] = neighbor_nodes[indices].tolist()
        neighbor_weights_dict[i] = sorted_weights.tolist()

    pickle.dump(neighbor_dict, open(neighbor_dict_path, "wb"))
    pickle.dump(neighbor_weights_dict, open(weights_dict_path, "wb"))
    print(f"[{mode}] Finish {target} Neighbor dict \nDict Path: {neighbor_dict_path}, \nWeight Dict: {weights_dict_path}")


def prepare_entity_graph(cfg, mode='train'):
    data_dir = {"train": cfg.dataset.train_dir, "val": cfg.dataset.val_dir, "test": cfg.dataset.test_dir}

    target_path = Path(data_dir[mode]) / "entity_graph.pt"
    reprocess_flag = False
    if target_path.exists() is False:
        reprocess_flag = True
    if (reprocess_flag == False) and (cfg.reprocess == False) and (cfg.reprocess_neighbors == False):
        print(f"[{mode}] Entity graph exists!")
        return

    entity_dict = pickle.load(open(Path(data_dir[mode]) / "entity_dict.bin", "rb"))
    origin_graph_path = Path(data_dir['train']) / "entity_graph.pt"

    if mode == 'train':
        target_news_graph_path = Path(data_dir[mode]) / "nltk_news_graph.pt"
        news_graph = torch.load(target_news_graph_path)
        entity_indices = news_graph.x[:, -8:-3].numpy()

        entity_edge_index = []

        news_edge_src, news_edge_dest = news_graph.edge_index
        edge_weights = news_graph.edge_attr.long().tolist()
        for i in range(news_edge_src.shape[0]):
            src_entities = entity_indices[news_edge_src[i]]
            dest_entities = entity_indices[news_edge_dest[i]]
            src_entities_mask = src_entities > 0
            dest_entities_mask = dest_entities > 0
            src_entities = src_entities[src_entities_mask]
            dest_entities = dest_entities[dest_entities_mask]
            edges = list(itertools.product(src_entities, dest_entities)) * edge_weights[i]
            entity_edge_index.extend(edges)

        edge_weights = Counter(entity_edge_index)
        unique_edges = list(edge_weights.keys())

        edge_index = torch.tensor(list(zip(*unique_edges)), dtype=torch.long)
        edge_attr = torch.tensor([edge_weights[edge] for edge in unique_edges], dtype=torch.long)

        edge_index, edge_attr = to_undirected(edge_index, edge_attr)

        data = Data(x=torch.arange(len(entity_dict) + 1), edge_index=edge_index, edge_attr=edge_attr, num_nodes=len(entity_dict) + 1)

        torch.save(data, target_path)
        print(f"[{mode}] Finish Entity Graph Construction, \n Graph Path: {target_path} \nGraph Info: {data}")
    elif mode in ['val', 'test']:
        origin_graph = torch.load(origin_graph_path)
        edge_index = origin_graph.edge_index
        edge_attr = origin_graph.edge_attr

        data = Data(x=torch.arange(len(entity_dict) + 1), edge_index=edge_index, edge_attr=edge_attr, num_nodes=len(entity_dict) + 1)

        torch.save(data, target_path)
        print(f"[{mode}] Finish Entity Graph Construction, \n Graph Path: {target_path} \nGraph Info: {data}")


def prepare_preprocessed_data(cfg):
    prepare_distributed_data(cfg, "train")
    prepare_distributed_data(cfg, "val")

    prepare_preprocess_bin(cfg, "train")
    prepare_preprocess_bin(cfg, "val")
    # prepare_preprocess_bin(cfg, "test")

    prepare_news_graph(cfg, 'train')
    prepare_news_graph(cfg, 'val')
    # prepare_news_graph(cfg, 'test')

    prepare_neighbor_list(cfg, 'train', 'news')
    prepare_neighbor_list(cfg, 'val', 'news')
    # prepare_neighbor_list(cfg, 'test', 'news')

    prepare_entity_graph(cfg, 'train')
    prepare_entity_graph(cfg, 'val')
    # prepare_entity_graph(cfg, 'test')

    prepare_neighbor_list(cfg, 'train', 'entity')
    prepare_neighbor_list(cfg, 'val', 'entity')
    # prepare_neighbor_list(cfg, 'test', 'entity')

    data_dir = {"train": cfg.dataset.train_dir, "val": cfg.dataset.val_dir, "test": cfg.dataset.test_dir}
    train_entity_emb_path = Path(data_dir['train']) / "entity_embedding.vec"
    val_entity_emb_path = Path(data_dir['val']) / "entity_embedding.vec"
    # test_entity_emb_path = Path(data_dir['test']) / "entity_embedding.vec"

    val_combined_path = Path(data_dir['val']) / "combined_entity_embedding.vec"
    # test_combined_path = Path(data_dir['test']) / "combined_entity_embedding.vec"

    # os.system("cat " + f"{train_entity_emb_path}" + f" > {val_combined_path}")
    os.system("cat " + f"{train_entity_emb_path} {val_entity_emb_path}" + f" > {val_combined_path}")
    # os.system("cat " + f"{train_entity_emb_path} {test_entity_emb_path}" + f" > {test_combined_path}")
    print("DONE!!!!")
