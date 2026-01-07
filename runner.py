"""PyTorch Lightning training module."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.metrics import f1_score, roc_auc_score

from fds import FDS
from optimizers import RAdam
import config


class Runner(pl.LightningModule):
    """PyTorch Lightning module for training CLIPDR."""
    
    def __init__(self, model, num_ranks=5):
        super().__init__()
        self.model = model
        self.num_ranks = num_ranks
        
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction="sum")
        
        self.FDS = FDS(
            feature_dim=config.FDS_FEATURE_DIM,
            bucket_num=config.FDS_BUCKET_NUM,
            bucket_start=config.FDS_BUCKET_START,
            start_update=config.FDS_START_UPDATE,
            start_smooth=config.FDS_START_SMOOTH,
            kernel=config.FDS_KERNEL,
            ks=config.FDS_KS,
            sigma=config.FDS_SIGMA,
            momentum=config.FDS_MOMENTUM
        )
        
        self.register_buffer(
            "rank_output_value_array",
            torch.arange(0, num_ranks).float(),
            persistent=False
        )
    
    def forward(self, x):
        return self.model(x)
    
    def run_step(self, batch, batch_idx, M):
        """
        Run one step of training/validation/testing.
        
        Args:
            batch: Input batch (images, labels)
            batch_idx: Batch index
            M: Mode (0 for train with rank loss, 1 for val/test without rank loss)
        """
        x, y = batch
        
        our_logits, image_features, text_features = self.model(x)
        our_logits = our_logits.float()
        
        if M == 0:
            rank_loss = self.rank_loss(our_logits, y)
            loss_kl = self.compute_kl_loss(our_logits, y)
            loss_ce = self.ce_loss(our_logits, y)
            loss = loss_ce + loss_kl + rank_loss
        else:
            loss_kl = self.compute_kl_loss(our_logits, y)
            loss_ce = self.ce_loss(our_logits, y)
            loss = loss_ce + loss_kl
        
        metrics_exp = self.compute_per_example_metrics(our_logits, y, "exp")
        metrics_max = self.compute_per_example_metrics(our_logits, y, "max")
        
        return {"loss": loss, **metrics_exp, **metrics_max}
    
    def training_step(self, batch, batch_idx):
        outputs = self.run_step(batch, batch_idx, M=0)
        self.logging(outputs, "train", on_step=True, on_epoch=True)
        return outputs
    
    def validation_step(self, batch, batch_idx):
        outputs = self.run_step(batch, batch_idx, M=1)
        self.logging(outputs, "val", on_step=False, on_epoch=True)
        return outputs
    
    def test_step(self, batch, batch_idx):
        outputs = self.run_step(batch, batch_idx, M=0)
        self.logging(outputs, "test", on_step=False, on_epoch=True)
        return outputs
    
    def compute_kl_loss(self, logits, y):
        """Compute KL divergence loss."""
        y_t = F.one_hot(y, self.num_ranks).t()
        y_t_row_ind = y_t.sum(-1) > 0
        num_slots = y_t_row_ind.sum()
        y_t_reduction = (y_t * 10.0).softmax(-1)
        y_t_reduction[y_t_row_ind <= 0] = 0
        logits_t = logits.t()
        return self.kl_loss(F.log_softmax(logits_t, dim=-1), y_t_reduction) / num_slots
    
    def rank_loss(self, our_logits, y):
        """Compute ranking loss for ordinal classification."""
        indexA = torch.nonzero(y == 0, as_tuple=True)[0]
        indexB = torch.nonzero(y == 1, as_tuple=True)[0]
        indexC = torch.nonzero(y == 2, as_tuple=True)[0]
        indexD = torch.nonzero(y == 3, as_tuple=True)[0]
        indexF = torch.nonzero(y == 4, as_tuple=True)[0]
        
        images_similarity1 = torch.zeros(len(y), 5, device=our_logits.device)
        images_similarity2 = torch.zeros(len(y), 5, device=our_logits.device)
        images_similarity3 = torch.zeros(len(y), 5, device=our_logits.device)
        
        logits_similarity_image1 = torch.zeros_like(images_similarity1)
        logits_similarity_image2 = torch.zeros_like(images_similarity2)
        logits_similarity_image3 = torch.zeros_like(images_similarity3)
        
        for index in indexA:
            images_similarity1[index, 0] = 1
            logits_similarity_image1[index, :2] = our_logits[index, :2]
            logits_similarity_image2[index, 1:3] = our_logits[index, 1:3]
            logits_similarity_image3[index, 2:4] = our_logits[index, 2:4]
        
        for index in indexB:
            images_similarity1[index, 1] = 1
            logits_similarity_image1[index, 1:3] = our_logits[index, 1:3]
            logits_similarity_image2[index, 2:4] = our_logits[index, 2:4]
            logits_similarity_image3[index, 3:5] = our_logits[index, 3:5]
        
        for index in indexC:
            images_similarity1[index, 2] = 1
            logits_similarity_image1[index, 2:4] = our_logits[index, 2:4]
            logits_similarity_image2[index, 3:5] = our_logits[index, 3:5]
        
        for index in indexD:
            images_similarity1[index, 3] = 1
            logits_similarity_image1[index, 3:5] = our_logits[index, 3:5]
        
        for index in indexF:
            images_similarity1[index, 4] = 1
            logits_similarity_image1[index, 4] = our_logits[index, 4]
        
        loss1 = nn.CrossEntropyLoss()(logits_similarity_image1, images_similarity1)
        loss2 = nn.CrossEntropyLoss()(logits_similarity_image2, images_similarity2)
        loss3 = nn.CrossEntropyLoss()(logits_similarity_image3, images_similarity3)
        
        return loss1 + loss2 + loss3
    
    def compute_per_example_metrics(self, logits, y, gather_type="exp"):
        """Compute metrics for current batch."""
        probs = F.softmax(logits, -1)
        dtype = logits.dtype
        
        if gather_type == "exp":
            predict_y = torch.sum(
                probs * self.rank_output_value_array.type(dtype),
                dim=-1
            )
        else:
            predict_y = torch.argmax(probs, dim=-1).type(dtype)
        
        mae = torch.abs(predict_y - y)
        acc = (torch.round(predict_y) == y).type(dtype)
        
        auc_ovo = roc_auc_score(
            y.cpu().numpy(),
            probs.detach().cpu().numpy(),
            average='macro',
            multi_class='ovo',
            labels=[0, 1, 2, 3, 4]
        )
        auc_ovo = torch.tensor(auc_ovo)
        
        f1 = f1_score(
            y.cpu().numpy(),
            torch.round(predict_y).detach().cpu().numpy(),
            average='macro'
        )
        f1 = torch.tensor(f1)
        
        return {
            f"mae_{gather_type}_metric": mae,
            f"acc_{gather_type}_metric": acc,
            f"{gather_type}_DGDR_auc_metric": auc_ovo,
            f"{gather_type}_DGDR_f1_metric": f1
        }
    
    def logging(self, outputs, run_type, on_step=True, on_epoch=True):
        """Log metrics."""
        for k, v in outputs.items():
            if k.endswith("metric") or k.endswith("loss"):
                self.log(
                    f"{run_type}_{k}",
                    v.mean(),
                    on_step=on_step,
                    on_epoch=on_epoch,
                    prog_bar=True,
                    logger=True
                )
    
    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        params = [
            {"params": self.model.prompt_learner.context_embeds, "lr": config.LEARNING_RATE},
            {"params": self.model.prompt_learner.rank_embeds, "lr": config.LEARNING_RATE},
            {"params": self.model.image_encoder.parameters(), "lr": config.LEARNING_RATE},
        ]
        
        optimizer = RAdam(
            params,
            lr=config.LEARNING_RATE,
            betas=config.BETAS,
            weight_decay=config.WEIGHT_DECAY,
            degenerated_to_sgd=False
        )
        
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=config.MILESTONES,
            gamma=config.GAMMA
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1
            }
        }