python pipelines/train.py \
--config-name=config.yaml \
job_id=QH9 \
data_name=escflow_qh9stable \
dataset_path="/data/qhflow-mlff/dataset/QH9Stable_shard/processed/lmdbs" \
model_backbone=QHNet_backbone \
wandb.wandb_api_key=$WANDB_API_KEY \
lr=0.0005 \
enable_hami=True \
hami_weight=1 \
devices=[0,7] \
batch_size=32 \
inference_batch_size=224 \
schedule=polynomial \
schedule.lr_warmup_steps=1000 \
max_steps=300000 \
train_ratio=0.8 \
val_ratio=0.1 \
test_ratio=0.1  \
gradient_clip_val=5.0 \
dataset_size=-1


# python pipelines/train.py --config-name=config.yaml wandb.open=True wandb.wandb_group="QH9" job_id=QHNet_SO2 wandb.wandb_api_key=6f1080f993d5d7ad6103e69ef57dd9291f1bf366 \
# dataset_path="/gpfs/gibbs/pi/gerstein/yl2428/qh9/QH9_new.db" model_backbone=QHNetBackBoneSO2 ngpus=4 lr=0.0005 enable_hami=True \
# hami_weight=1  batch_size=32 schedule=polynomial schedule.lr_warmup_steps=1000 \
# max_steps=300000 used_cache=True \
# train_ratio=0.9 val_ratio=0.06 test_ratio=0.04  gradient_clip_val=5.0 dataset_size=100000