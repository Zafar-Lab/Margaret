import torch
import torch.nn as nn


class MaskedBCEWithLogitsLoss(nn.Module):
    def __init__(self, ignore_index=-1, pos_weight=None, minimize_entropy=False):
        super(MaskedBCEWithLogitsLoss, self).__init__()
        self.minimize_entropy = minimize_entropy
        self.ignore_index = ignore_index
        self.pos_weight = None
        # self.pos_weight = pos_weight or torch.tensor([0.62, 0.83, 0.31, 0.41, 0.96, 0.25, 0.78, 0.21, 0.25, 0.14, 0.27])
        self.loss = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight, reduction='none')

    def forward(self, preds, target):
        assert preds.shape == target.shape
        # Ignore predictions with label ignore_index
        mask = (target != self.ignore_index)
        loss = torch.mean(self.loss(preds, target) * mask)
        if self.minimize_entropy:
            sig_pred = torch.sigmoid(preds)
            entropy_loss = torch.mean(-sig_pred * torch.log(sig_pred) * mask)
            loss = loss + entropy_loss
        return loss


class OhemMaskedBCEWithLogitsLoss(nn.Module):
    def __init__(self, ignore_index=-1, minimize_entropy=False):
        super(OhemMaskedBCEWithLogitsLoss, self).__init__()
        self.minimize_entropy = minimize_entropy
        self.ignore_index = ignore_index
        self.hepc = 15
        # Obtained using median frequency balancing
        # self.pos_weight = torch.Tensor([0.78, 0.89, 0.64, 0.69, 0.94, 0.62, 0.87, 0.6, 0.62, 0.57, 0.63])
        # self.neg_weight = torch.Tensor([1.38, 1.15, 2.24, 1.81, 1.06, 2.54, 1.18, 3.0, 2.59, 4.2, 2.45])
        # Inverse frequency weighted weights
        # self.pos_weight = torch.Tensor([0.38, 0.45, 0.24, 0.29, 0.49, 0.2, 0.44, 0.17, 0.2, 0.13, 0.21])
        # self.neg_weight = torch.Tensor([0.62, 0.55, 0.76, 0.71, 0.51, 0.8, 0.56, 0.83, 0.8, 0.87, 0.79])
        self.epsilon = 1e-8

    def forward(self, logits, target):
        assert logits.shape == target.shape
        pred_probs = torch.sigmoid(logits)
        num_classes = pred_probs.shape[-1]
        batch_size = pred_probs.shape[0]
        loss = 0
        for cls in range(num_classes):
            cls_preds = pred_probs[:, cls]
            cls_targets = target[:, cls]
            valid_inds = cls_targets != self.ignore_index
            preds_ = cls_preds[valid_inds]
            targets_ = cls_targets[valid_inds]
            cw_ce_loss = -(targets_ * torch.log(preds_ + self.epsilon) + \
                        (1 - targets_) * torch.log(1 - preds_ + self.epsilon))
            # Take some hard examples (with high loss values) per class and backprop only over those
            sorted_cw_ce_loss, idx = torch.sort(cw_ce_loss, descending=True)[:self.hepc]
            if self.hepc < sorted_cw_ce_loss.size(0):
                cw_ce_loss = cw_ce_loss[idx[:self.hepc]]
            mean_ce_loss = torch.mean(cw_ce_loss)
            loss += torch.mean(cw_ce_loss)
        bce_loss = loss / num_classes
        return bce_loss


class MaskedMarginLoss(nn.Module):
    def __init__(self):
        super(MaskedMarginLoss, self).__init__()
        self.ignore_index = -1
        self.epsilon = 1e-8
        self.criterion = nn.MultiLabelMarginLoss()

    def forward(self, logits, target):
        assert logits.shape == target.shape
        pred_probs = logits
        sorted_targets, inds = torch.sort(target, descending=True)
        sorted_targets = sorted_targets.long()
        perm_preds = pred_probs.gather(1, inds)
        return self.criterion(perm_preds, sorted_targets)


class MaskedMSELoss(nn.Module):
    def __init__(self, **kwargs):
        super(MaskedMSELoss, self).__init__()
        self.ignore_index = -1
        self.epsilon = 1e-8
        self.criterion = nn.MSELoss(reduction='none')

    def forward(self, logits, target):
        assert logits.shape == target.shape
        pred_probs = torch.sigmoid(logits)
        mask = (target != self.ignore_index)
        loss = torch.mean(self.criterion(pred_probs, target) * mask)
        return loss


class MaskedL1Loss(nn.Module):
    def __init__(self, **kwargs):
        super(MaskedMSELoss, self).__init__()
        self.ignore_index = -1
        self.epsilon = 1e-8
        self.criterion = nn.L1Loss(reduction='none')

    def forward(self, logits, target):
        assert logits.shape == target.shape
        pred_probs = torch.sigmoid(logits)
        mask = (target != self.ignore_index)
        loss = torch.mean(self.criterion(pred_probs, target) * mask)
        return loss
