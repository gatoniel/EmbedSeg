import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from EmbedSeg.criterions.lovasz_losses import lovasz_hinge
from EmbedSeg.criterions.my_loss import SpatialEmbLoss
from EmbedSeg.criterions.my_loss_3d import SpatialEmbLoss_3d


class SpatialEmbTrackingLoss(nn.Module):
    """
    Segmentation loss for the reference frame ``t`` (the unchanged
    ``SpatialEmbLoss`` / ``SpatialEmbLoss_3d`` on the first ``nd + n_sigma + 1``
    channels) plus a tracking loss on the following ``nd + n_sigma + 1`` channels:

    * ``nd`` offsets: ``tanh(offset) + grid`` is the predicted position, in frame
      ``t-1``, of the centre of the parent of the object the pixel belongs to
    * ``n_sigma`` sigma channels (same role as in segmentation)
    * 1 division logit

    The tracking targets are grouped by *parent*: all pixels of all daughters of
    one parent share one Gaussian around the parent centre and one Lovasz-hinge
    term over the union of the daughters (otherwise siblings, which point to the
    same centre, would push each other away).

    Samples with ``has_context == False`` (no previous frame) only contribute to
    the segmentation loss. Works for 2D (``grid_z=None``) and 3D.
    """

    def __init__(
        self,
        grid_z=None,
        grid_y=1024,
        grid_x=1024,
        pixel_z=None,
        pixel_y=1,
        pixel_x=1,
        one_hot=False,
        n_sigma=None,
        foreground_weight=1,
        div_pos_weight=5.0,
    ):
        super().__init__()
        self.ndim = 2 if grid_z is None else 3
        self.n_sigma = self.ndim if n_sigma is None else n_sigma
        self.div_pos_weight = div_pos_weight
        if self.ndim == 2:
            self.seg_loss = SpatialEmbLoss(
                grid_y, grid_x, pixel_y, pixel_x, one_hot, self.n_sigma, foreground_weight
            )
        else:
            self.seg_loss = SpatialEmbLoss_3d(
                grid_z,
                grid_y,
                grid_x,
                pixel_z,
                pixel_y,
                pixel_x,
                one_hot,
                self.n_sigma,
                foreground_weight,
            )
        self.n_seg = self.ndim + self.n_sigma + 1
        self.last_terms = {}
        self._metrics = self._empty_metrics()

    @staticmethod
    def _empty_metrics():
        return dict(correct=0, total=0, tp=0, fp=0, fn=0)

    @property
    def grid(self):
        return self.seg_loss.xym if self.ndim == 2 else self.seg_loss.xyzm

    @torch.no_grad()
    def _update_metrics(self, emb, div_logit, instance, parent, inst_prev):
        """Validation metrics on the GT segmentation of sample ``b``: does the
        predicted position point into the right parent (``associate_tracks``), and
        is the division flag right (object level, mean probability >= 0.5)?"""
        from EmbedSeg.utils.tracking_inference import (
            associate_tracks,
            embedding_to_index,
            grid_steps,
        )

        inst = instance[0].cpu().numpy().astype(np.int32)
        par = parent[0].cpu().numpy()
        prev = inst_prev[0].cpu().numpy().astype(np.int32)
        steps = grid_steps(self.grid.cpu().numpy())
        idx = embedding_to_index(emb.cpu().numpy(), steps)
        prob = torch.sigmoid(div_logit).cpu().numpy()
        res = associate_tracks(inst, idx, prev, division_prob=prob, next_id=int(max(prev.max(), inst.max())) + 1)
        gt_children = {}
        for i in np.unique(inst[inst > 0]):
            p = int(par[inst == i].max())
            gt_children.setdefault(p, []).append(int(i))
        for i, pred_p in res["parents"].items():
            gt_p = int(par[inst == i].max())
            if gt_p < 0:  # parent unknown (hidden / outside the crop): no label
                continue
            self._metrics["total"] += 1
            self._metrics["correct"] += int(pred_p == gt_p)
            gt_div = gt_p > 0 and len(gt_children[gt_p]) > 1
            pred_div = bool(prob[inst == i].mean() >= 0.5)
            self._metrics["tp"] += int(gt_div and pred_div)
            self._metrics["fp"] += int((not gt_div) and pred_div)
            self._metrics["fn"] += int(gt_div and not pred_div)

    def report_tracking_metrics(self, reset=True):
        """Association accuracy and division F1 accumulated since the last reset
        (collected while the loss is called with ``iou=True``, i.e. in validation)."""
        m = self._metrics
        tp, fp, fn = m["tp"], m["fp"], m["fn"]
        out = dict(
            assoc_acc=m["correct"] / m["total"] if m["total"] else 0.0,
            n_objects=m["total"],
            div_f1=2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0,
        )
        if reset:
            self._metrics = self._empty_metrics()
        return out

    def forward(
        self,
        prediction,
        instances,
        labels,
        center_images,
        instances_prev=None,
        center_images_prev=None,
        parents=None,
        divisions=None,
        has_context=None,
        w_inst=1,
        w_var=10,
        w_seed=1,
        w_track=1,
        w_track_reg=1,
        w_div=1,
        iou=False,
        iou_meter=None,
    ):
        nd, ns, o = self.ndim, self.n_sigma, self.n_seg
        seg = self.seg_loss(
            prediction[:, :o],
            instances,
            labels,
            center_images,
            w_inst=w_inst,
            w_var=w_var,
            w_seed=w_seed,
            iou=iou,
            iou_meter=iou_meter,
        )
        self.last_terms = {"seg": float(seg.detach())}
        if prediction.size(1) < 2 * o or parents is None:
            return seg  # no tracking head / targets: pure segmentation

        batch_size = prediction.size(0)
        spatial = prediction.shape[2:]
        grid = self.grid[(slice(None),) + tuple(slice(0, s) for s in spatial)].contiguous()
        ones = (1,) * nd
        pos_weight = torch.tensor(
            [self.div_pos_weight], device=prediction.device, dtype=prediction.dtype
        )

        track_total = 0
        n_ctx = 0
        terms = dict(track=0.0, var=0.0, reg=0.0, div=0.0)
        for b in range(batch_size):
            if has_context is not None and not bool(has_context[b]):
                continue
            n_ctx += 1
            emb = torch.tanh(prediction[b, o : o + nd]) + grid  # nd x sp
            sigma = prediction[b, o + nd : o + nd + ns]  # ns x sp
            div_logit = prediction[b, o + nd + ns]  # sp

            instance = instances[b].unsqueeze(0)  # 1 x sp
            parent = parents[b].unsqueeze(0)
            inst_prev = instances_prev[b].unsqueeze(0)
            cen_prev = center_images_prev[b].unsqueeze(0)
            ignore = instance < 0

            parent_ids = parent.unique()
            parent_ids = parent_ids[parent_ids > 0]
            inst_l, var_l, reg_l, count = 0, 0, 0, 0
            for p in parent_ids:
                group = (parent == p) & (instance > 0)  # all daughters of p
                prev_mask = inst_prev == p
                if group.sum() == 0 or prev_mask.sum() == 0:
                    continue
                center_mask = prev_mask & cen_prev
                if center_mask.sum().eq(1):
                    center = grid[center_mask.expand_as(grid)].view(nd, *ones)
                else:
                    center = grid[prev_mask.expand_as(grid)].view(nd, -1).mean(1).view(nd, *ones)

                sigma_in = sigma[group.expand_as(sigma)].view(ns, -1)
                s = sigma_in.mean(1).view(ns, 1)
                var_l = var_l + torch.mean(torch.pow(sigma_in - s.detach(), 2))

                sq = torch.pow(emb - center, 2)  # nd x sp
                reg_l = reg_l + sq.sum(0)[group[0]].mean()

                s = torch.exp(s.view(ns, *ones) * 10)
                dist = torch.exp(-torch.sum(sq * s, 0, keepdim=True))  # 1 x sp
                lovasz_labels = group.long()
                lovasz_labels[ignore] = -1
                inst_l = inst_l + lovasz_hinge(dist * 2 - 1, lovasz_labels, ignore=-1)
                count += 1

            sample_loss = 0
            if count > 0:
                inst_l, var_l, reg_l = inst_l / count, var_l / count, reg_l / count
                sample_loss = w_track * inst_l + w_var * var_l + w_track_reg * reg_l
                terms["track"] += float(inst_l.detach())
                terms["var"] += float(var_l.detach())
                terms["reg"] += float(reg_l.detach())

            if iou:  # validation: collect association / division metrics
                self._update_metrics(emb, div_logit, instance, parent, inst_prev)

            # division label only where the parent is known (parent == -1: unknown)
            fg = (instance[0] > 0) & (parent[0] >= 0)
            if fg.any():
                div = F.binary_cross_entropy_with_logits(
                    div_logit[fg],
                    divisions[b][fg].to(div_logit.dtype),
                    pos_weight=pos_weight,
                )
                sample_loss = sample_loss + w_div * div
                terms["div"] += float(div.detach())
            track_total = track_total + sample_loss

        if n_ctx > 0:
            track_total = track_total / n_ctx
            self.last_terms.update({k: v / n_ctx for k, v in terms.items()})
        return seg + track_total + prediction.sum() * 0
