#!/usr/bin/env python
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from src.benchmark import L3SFV2AugmentedBenchmark
from src.gmdataset import GMDataset, get_dataloader
from src.model.ngm import Net
from src.model.match_classifier import MatchClassifier
from utils.data_to_cuda import data_to_cuda
from utils.models_sl import load_model, save_model

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Hyperparameters
DATASET_LEN = 640
LR = 1e-4
NUM_EPOCHS = 5
PRETRAINED_PATH = "results/base/params/best_model.pt"

# Dataset and dataloader
benchmark = L3SFV2AugmentedBenchmark(
    sets='train',
    obj_resize=(320, 240),
    train_root='/green/data/L3SF_V2/L3SF_V2_Augmented',
)

dataset = GMDataset("L3SFV2Augmented", benchmark, DATASET_LEN, True, None, "2GM")
dataloader = get_dataloader(dataset, shuffle=True, fix_seed=False)

# Pretrained matching network
match_net = Net(regression=False)
if Path(PRETRAINED_PATH).exists():
    logger.info("Loading pretrained network from %s", PRETRAINED_PATH)
    load_model(match_net, PRETRAINED_PATH)
else:
    logger.warning("Pretrained model not found at %s", PRETRAINED_PATH)

for p in match_net.parameters():
    p.requires_grad = False
match_net.eval()

# Match classifier to train
classifier = MatchClassifier()

criterion = nn.BCELoss()
optimizer = optim.Adam(classifier.parameters(), lr=LR)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
match_net.to(device)
classifier.to(device)

checkpoint_dir = Path("result/match_classifier")
checkpoint_dir.mkdir(parents=True, exist_ok=True)

for epoch in range(NUM_EPOCHS):
    classifier.train()
    epoch_loss = 0.0
    for batch in dataloader:
        batch = data_to_cuda(batch)
        with torch.no_grad():
            outputs = match_net(batch, regression=False)
        sim = outputs['ds_mat']
        corr = outputs['perm_mat']
        pred = classifier(sim, corr)
        # label 1 if same class else 0
        labels = (batch['cls'][0] == batch['cls'][1]).float().unsqueeze(1).to(pred.device)
        loss = criterion(pred, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    avg_loss = epoch_loss / len(dataloader)
    logger.info("Epoch %d average loss %.4f", epoch, avg_loss)
    save_model(classifier, str(checkpoint_dir / f'classifier_{epoch+1:04}.pt'))

logger.info("Training complete. Final classifier saved to %s", checkpoint_dir)
