
# Testing
import os.path
from pathlib import Path

import hydra
import wandb
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.cuda import amp
import torch.multiprocessing as mp
import torch.distributed as dist
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR

from dataload.data_load import load_data
from dataload.data_preprocess import prepare_preprocessed_data
from utils.metrics import *
from utils.common import *

### custom your wandb setting here ###
# os.environ["WANDB_API_KEY"] = ""
# os.environ["WANDB_MODE"] = "offline"

def train(model, optimizer, scaler, scheduler, dataloader, local_rank, cfg, early_stopping):
    model.train()
    torch.set_grad_enabled(True)

    sum_loss = torch.zeros(1).to(local_rank)
    sum_auc = torch.zeros(1).to(local_rank)
    
    # validation
    res = val(model, local_rank, cfg)
    model.train()

    if local_rank == 0:
        # save_model(cfg, model, optimizer, f"{cfg.ml_label}_auc_best")
        wandb.run.summary.update({"best_auc": res["auc"],"best_mrr":res['mrr'], 
                            "best_ndcg5":res['ndcg5'], "best_ndcg10":res['ndcg10']})

        pretty_print(res)
        wandb.log(res)



def val(model, local_rank, cfg):
    model.eval()
    dataloader = load_data(cfg, mode='val', model=model, local_rank=local_rank)
    tasks = []
    with torch.no_grad():
        for cnt, (subgraph, mappings, clicked_entity, candidate_input, candidate_entity, entity_mask, labels) \
                in enumerate(tqdm(dataloader,
                                  total=int(len(dataloader)),
                                  desc=f"[{local_rank}] Validating")):
            candidate_emb = torch.FloatTensor(np.array(candidate_input)).to(local_rank, non_blocking=True)
            candidate_entity = candidate_entity.to(local_rank, non_blocking=True)
            entity_mask = entity_mask.to(local_rank, non_blocking=True)
            clicked_entity = clicked_entity.to(local_rank, non_blocking=True)

            scores = model.module.validation_process(subgraph, mappings, clicked_entity, candidate_emb, candidate_entity, entity_mask)
            
            tasks.append((labels.tolist(), scores))

    with mp.Pool(processes=cfg.num_workers) as pool:
        results = pool.map(cal_metric, tasks)
    val_auc, val_mrr, val_ndcg5, val_ndcg10 = np.array(results).T


    # barrier
    torch.distributed.barrier()

    reduced_auc = reduce_mean(torch.tensor(np.nanmean(val_auc)).float().to(local_rank), cfg.gpu_num)
    reduced_mrr = reduce_mean(torch.tensor(np.nanmean(val_mrr)).float().to(local_rank), cfg.gpu_num)
    reduced_ndcg5 = reduce_mean(torch.tensor(np.nanmean(val_ndcg5)).float().to(local_rank), cfg.gpu_num)
    reduced_ndcg10 = reduce_mean(torch.tensor(np.nanmean(val_ndcg10)).float().to(local_rank), cfg.gpu_num)

    res = {
        "auc": reduced_auc.item(),
        "mrr": reduced_mrr.item(),
        "ndcg5": reduced_ndcg5.item(),
        "ndcg10": reduced_ndcg10.item(),
    }
    
    return res

def main_worker(local_rank, cfg):
    # -----------------------------------------Environment Initial
    seed_everything(cfg.seed)
    dist.init_process_group(backend='nccl',
                            init_method='tcp://127.0.0.1:23456',
                            world_size=cfg.gpu_num,
                            rank=local_rank)

    # -----------------------------------------Dataset & Model Load
    num_training_steps = int(cfg.num_epochs * cfg.dataset.pos_count / (cfg.batch_size * cfg.accumulation_steps))
    num_warmup_steps = int(num_training_steps * cfg.warmup_ratio + 1)
    train_dataloader = load_data(cfg, mode='train', local_rank=local_rank)
    model = load_model(cfg).to(local_rank)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.optimizer.lr)

    lr_lambda = lambda step: 1.0 if step > num_warmup_steps else step / num_warmup_steps
    scheduler = LambdaLR(optimizer, lr_lambda)
    
    # ------------------------------------------Load Checkpoint & optimizer
    if cfg.load_checkpoint:
        file_path = Path(f"{cfg.path.ckp_dir}/{cfg.model.model_name}_{cfg.dataset.dataset_name}_{cfg.load_mark}.pth")
        checkpoint = torch.load(file_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])  # After Distributed
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    optimizer.zero_grad(set_to_none=True)
    scaler = amp.GradScaler()

    # ------------------------------------------Main Start
    early_stopping = EarlyStopping(cfg.early_stop_patience)

    if local_rank == 0:
        wandb.init(config=OmegaConf.to_container(cfg, resolve=True),
                   project=cfg.logger.exp_name, name=cfg.logger.run_name)
        print(model)

    val(model, local_rank, cfg)


    if local_rank == 0:
        wandb.finish()


@hydra.main(version_base="1.2", config_path=os.path.join(get_root(), "configs"), config_name="EB")
def main(cfg: DictConfig):
    seed_everything(cfg.seed)
    cfg.gpu_num = torch.cuda.device_count()
    # prepare_preprocessed_data(cfg)
    mp.spawn(main_worker, nprocs=cfg.gpu_num, args=(cfg,))


if __name__ == "__main__":
    main()

