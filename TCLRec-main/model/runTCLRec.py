import sys
import logging
from logging import getLogger
from recbole.utils import init_logger, init_seed, set_color
from recbole.trainer import Trainer
from TCLRec import TCLRec
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.data.dataloader.general_dataloader import FullSortEvalDataLoader
from recbole.data.interaction import Interaction
import dgl
import torch
import heapq
from collections import defaultdict, Counter
from scipy.sparse import coo_matrix
from scipy.stats import pearsonr
import time
import pandas as pd
import numpy as np

# ---------- 图构建 ----------
def build_item_similarity_graph(inter_feat, num_items, device, top_k=2, alpha=0.2):
    print("=== 开始构建物品相似度图 ===")
    start_time = time.time()
    user_item_rating = defaultdict(dict)
    item_count = defaultdict(int)
    item_user_list = defaultdict(set)

    for uid, iid, rating in zip(inter_feat['user_id'], inter_feat['item_id'], inter_feat['rating']):
        uid, iid = int(uid), int(iid)
        user_item_rating[uid][iid] = float(rating)
        item_count[iid] += 1
        item_user_list[iid].add(uid)

    items = list(item_count.keys())
    min_pop, max_pop = min(item_count.values()), max(item_count.values())
    nor_pop = {i: (item_count[i] - min_pop) / (max_pop - min_pop + 1e-8) for i in items}

    row_indices, col_indices, data = [], [], []

    for i in items:
        sims = []
        candidate_items = set()
        for user in item_user_list[i]:
            candidate_items.update(user_item_rating[user].keys())
        candidate_items.discard(i)

        for j in candidate_items:
            common_users = item_user_list[i] & item_user_list[j]
            if not common_users:
                continue
            pop_i, pop_j = nor_pop[i], nor_pop[j]
            pop_bias = abs(pop_i - pop_j)
            w_i = 1 if pop_i < alpha else 1 / (pop_i * pop_bias + 1e-8)
            w_j = 1 if pop_j < alpha else 1 / (pop_j * pop_bias + 1e-8)
            num, denom_i, denom_j = 0.0, 0.0, 0.0
            for u in common_users:
                y_ui = user_item_rating[u][i]
                y_uj = user_item_rating[u][j]
                num += w_i * y_ui * w_j * y_uj
                denom_i += (w_i * y_ui) ** 2
                denom_j += (w_j * y_uj) ** 2
            sim = num / (denom_i ** 0.5 * denom_j ** 0.5 + 1e-8)
            if sim > 0:
                sims.append((sim, j))
        for sim, j in heapq.nlargest(top_k, sims):
            row_indices.extend([i, j])
            col_indices.extend([j, i])
            data.extend([sim, sim])

    sparse_matrix = coo_matrix((data, (row_indices, col_indices)), shape=(num_items, num_items))
    g = dgl.from_scipy(sparse_matrix, device=device)
    g = dgl.add_self_loop(g)
    num_self_loops = g.num_edges() - len(data)
    data.extend([1.0] * num_self_loops)
    g.edata['weight'] = torch.tensor(data, dtype=torch.float32, device=device)
    print(f"=== 物品相似度图构建完成，耗时 {time.time() - start_time:.2f} 秒 ===")
    return g

# ---------- 主程序 ----------
if __name__ == '__main__':
    config = Config(model=TCLRec, config_file_list=['./model/TCLRec.yaml'])
    init_seed(config['seed'], config['reproducibility'])

    init_logger(config)
    logger = getLogger()
    logger.info(sys.argv)
    logger.info(config)

    dataset = create_dataset(config)
    logger.info(dataset)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])
    device = config['device']

    print("构建物品相似度图开始...")
    graph = build_item_similarity_graph(train_data.dataset.inter_feat, dataset.item_num, device)

    model = TCLRec(config, train_data.dataset, graph, train_data.dataset.inter_feat).to(device)
    logger.info(model)

    trainer = Trainer(config, model)
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data, show_progress=config["show_progress"])
    test_result = trainer.evaluate(test_data, show_progress=config["show_progress"])

    logger.info(set_color("best valid ", "yellow") + f": {best_valid_result}")
    logger.info(set_color("test result", "yellow") + f": {test_result}")

