import torch
import torch.nn as nn
from EmbedSeg.criterions.lovasz_losses import lovasz_hinge


class SpatialEmbLoss_3d(nn.Module):
    def __init__(
        self,
        grid_z=32,
        grid_y=1024,
        grid_x=1024,
        pixel_z=1,
        pixel_y=1,
        pixel_x=1,
        one_hot=False,
        n_sigma=3,
        foreground_weight=1,
        memory_efficient=True,
    ):
        super().__init__()
        self.memory_efficient = memory_efficient

        print(
            "Created spatial emb loss function with: \
                    n_sigma: {}, foreground_weight: {}".format(
                n_sigma, foreground_weight
            )
        )
        print("*************************")
        self.n_sigma = n_sigma
        self.foreground_weight = foreground_weight

        xm = (
            torch.linspace(0, pixel_x, grid_x)
            .view(1, 1, 1, -1)
            .expand(1, grid_z, grid_y, grid_x)
        )
        ym = (
            torch.linspace(0, pixel_y, grid_y)
            .view(1, 1, -1, 1)
            .expand(1, grid_z, grid_y, grid_x)
        )
        zm = (
            torch.linspace(0, pixel_z, grid_z)
            .view(1, -1, 1, 1)
            .expand(1, grid_z, grid_y, grid_x)
        )
        xyzm = torch.cat((xm, ym, zm), 0)

        self.register_buffer("xyzm", xyzm)
        self.one_hot = one_hot

    def forward(self, prediction, *args, **kwargs):
        """Dispatch between the memory-efficient and the reference loss.

        With ``memory_efficient`` (default) and gradients enabled, every
        instance's loss term is back-propagated as soon as it is computed, so
        autograd graph memory no longer grows with the number of instances
        (see ``_forward_per_instance_backward``). Values and gradients are
        identical to ``_forward_dense``.
        """
        if self.memory_efficient and torch.is_grad_enabled() and prediction.requires_grad:
            return self._forward_per_instance_backward(prediction, *args, **kwargs)
        return self._forward_dense(prediction, *args, **kwargs)

    def _forward_per_instance_backward(
        self,
        prediction,
        instances,
        labels,
        center_images,
        w_inst=1,
        w_var=10,
        w_seed=1,
        iou=False,
        iou_meter=None,
    ):
        """Same loss as ``_forward_dense`` with O(1) graph memory in #instances.

        The dense version keeps several full-patch tensors per instance alive
        until one final ``backward()``. Here ``spatial_emb``, ``sigma`` and
        ``seed_map`` are detached into leaf tensors; each instance's term is
        scaled by its final weight (1/obj_count, 1/n_valid, 1/batch_size) and
        back-propagated immediately, which frees its graph and accumulates the
        gradient on the leaves. A surrogate term ``sum(leaf_value * grad)``
        then routes the accumulated gradients to ``prediction`` when the
        caller runs ``loss.backward()``. It is subtracted by its detached
        value so the returned loss value is the true loss.
        """
        batch_size, depth, height, width = (
            prediction.size(0),
            prediction.size(2),
            prediction.size(3),
            prediction.size(4),
        )
        xyzm_s = self.xyzm[:, 0:depth, 0:height, 0:width].contiguous()  # 3 x d x h x w

        total = prediction.new_zeros(())  # accumulated loss value, no graph
        surrogate = prediction.new_zeros(())  # carries gradient to prediction

        for b in range(batch_size):
            # differentiable (cheap, once per sample) view of the prediction
            spatial_emb_g = torch.tanh(prediction[b, 0:3]) + xyzm_s
            sigma_g = prediction[b, 3 : 3 + self.n_sigma]
            seed_map_g = torch.sigmoid(
                prediction[b, 3 + self.n_sigma : 3 + self.n_sigma + 1]
            )
            # leaves the per-instance graphs hang off
            spatial_emb = spatial_emb_g.detach().requires_grad_()
            sigma = sigma_g.detach().requires_grad_()
            seed_map = seed_map_g.detach().requires_grad_()

            instance = instances[b].unsqueeze(0)  # 1 x d x h x w
            label = labels[b].unsqueeze(0)
            center_image = center_images[b].unsqueeze(0)
            instance_ids = instance.unique()
            instance_ids = instance_ids[instance_ids > 0]
            obj_count = len(instance_ids)

            ignore = instance < 0
            n_valid = (~ignore).sum().clamp(min=1)
            bg_mask = (label == 0) & ~ignore
            # every term below is already multiplied by its final weight
            scale = 1.0 / batch_size

            with torch.enable_grad():
                if bg_mask.sum() > 0:
                    seed_bg = (
                        w_seed
                        * scale
                        * torch.sum(torch.pow(seed_map[bg_mask] - 0, 2))
                        / n_valid
                    )
                    seed_bg.backward()
                    total += seed_bg.detach()

                for id in instance_ids:
                    in_mask = instance.eq(id)
                    center_mask = in_mask & center_image
                    with torch.no_grad():
                        if center_mask.sum().eq(1):
                            center = xyzm_s[center_mask.expand_as(xyzm_s)].view(3, 1, 1, 1)
                        else:
                            xyz_in = xyzm_s[in_mask.expand_as(xyzm_s)].view(3, -1)
                            center = xyz_in.mean(1).view(3, 1, 1, 1)

                    sigma_in = sigma[in_mask.expand_as(sigma)].view(self.n_sigma, -1)
                    s = sigma_in.mean(1).view(self.n_sigma, 1, 1, 1)
                    var_loss = torch.mean(torch.pow(sigma_in - s[..., 0, 0].detach(), 2))

                    s = torch.exp(s * 10)
                    dist = torch.exp(
                        -1
                        * torch.sum(
                            torch.pow(spatial_emb - center, 2) * s, 0, keepdim=True
                        )
                    )

                    lovasz_labels = in_mask.long()
                    lovasz_labels[ignore] = -1
                    instance_loss = lovasz_hinge(
                        dist * 2 - 1, lovasz_labels, ignore=-1
                    )
                    seed_loss = (
                        self.foreground_weight
                        * torch.sum(
                            torch.pow(seed_map[in_mask] - dist[in_mask].detach(), 2)
                        )
                        / n_valid
                    )

                    term = scale * (
                        (w_inst * instance_loss + w_var * var_loss) / obj_count
                        + w_seed * seed_loss
                    )
                    term.backward()  # frees this instance's graph
                    total += term.detach()

                    if iou:
                        iou_meter.update(calculate_iou(dist.detach() > 0.5, in_mask))

            for g, leaf in ((spatial_emb_g, spatial_emb), (sigma_g, sigma), (seed_map_g, seed_map)):
                if leaf.grad is not None:
                    surrogate = surrogate + (g * leaf.grad).sum()

        # value == total; gradient wrt prediction == accumulated leaf grads.
        # `prediction.sum() * 0` of the dense path is covered: the surrogate
        # touches prediction, so unused-parameter bookkeeping (DDP) still works.
        return total + surrogate - surrogate.detach() + prediction.sum() * 0

    def _forward_dense(
        self,
        prediction,
        instances,
        labels,
        center_images,
        w_inst=1,
        w_var=10,
        w_seed=1,
        iou=False,
        iou_meter=None,
    ):
        # instances B 1 Z Y X
        batch_size, depth, height, width = (
            prediction.size(0),
            prediction.size(2),
            prediction.size(3),
            prediction.size(4),
        )

        xyzm_s = self.xyzm[:, 0:depth, 0:height, 0:width].contiguous()  # 3 x d x h x w

        loss = 0

        for b in range(0, batch_size):
            spatial_emb = torch.tanh(prediction[b, 0:3]) + xyzm_s  # 3 x d x h x w
            sigma = prediction[b, 3 : 3 + self.n_sigma]  # n_sigma x d x h x w
            seed_map = torch.sigmoid(
                prediction[b, 3 + self.n_sigma : 3 + self.n_sigma + 1]
            )  # 1 x d x h x w
            # loss accumulators
            var_loss = 0
            instance_loss = 0
            seed_loss = 0
            obj_count = 0

            instance = instances[b].unsqueeze(0)  # 1 x d x h x w
            label = labels[b].unsqueeze(0)  # 1 x d x h x w
            center_image = center_images[b].unsqueeze(0)  # 1 x d x h x w
            instance_ids = instance.unique()
            instance_ids = instance_ids[instance_ids > 0]

            # pixels labelled -1 carry no information and are excluded from the loss
            ignore = instance < 0  # 1 x d x h x w
            n_valid = (~ignore).sum().clamp(min=1)

            # regress bg to zero
            bg_mask = (label == 0) & ~ignore

            if bg_mask.sum() > 0:
                seed_loss += torch.sum(torch.pow(seed_map[bg_mask] - 0, 2))

            for id in instance_ids:
                in_mask = instance.eq(id)  # 1 x d x h x w
                # center_mask = in_mask & center_image.byte()
                center_mask = in_mask & center_image
                if center_mask.sum().eq(1):
                    center = xyzm_s[center_mask.expand_as(xyzm_s)].view(3, 1, 1, 1)
                else:
                    xyz_in = xyzm_s[in_mask.expand_as(xyzm_s)].view(3, -1)
                    center = xyz_in.mean(1).view(3, 1, 1, 1)  # 3 x 1 x 1 x 1

                # calculate sigma
                sigma_in = sigma[in_mask.expand_as(sigma)].view(
                    self.n_sigma, -1
                )  # 3 x N

                s = sigma_in.mean(1).view(self.n_sigma, 1, 1, 1)  # n_sigma x 1 x 1 x 1

                # calculate var loss before exp
                var_loss = var_loss + torch.mean(
                    torch.pow(sigma_in - s[..., 0, 0].detach(), 2)
                )

                s = torch.exp(s * 10)
                dist = torch.exp(
                    -1
                    * torch.sum(torch.pow(spatial_emb - center, 2) * s, 0, keepdim=True)
                )

                # apply lovasz-hinge loss
                lovasz_labels = in_mask.long()
                lovasz_labels[ignore] = -1
                instance_loss = instance_loss + lovasz_hinge(
                    dist * 2 - 1, lovasz_labels, ignore=-1
                )

                # seed loss
                seed_loss += self.foreground_weight * torch.sum(
                    torch.pow(seed_map[in_mask] - dist[in_mask].detach(), 2)
                )

                # calculate instance iou
                if iou:
                    iou_meter.update(calculate_iou(dist > 0.5, in_mask))

                obj_count += 1

            if obj_count > 0:
                instance_loss /= obj_count
                var_loss /= obj_count

            seed_loss = seed_loss / n_valid

            loss += w_inst * instance_loss + w_var * var_loss + w_seed * seed_loss

        loss = loss / (b + 1)

        return loss + prediction.sum() * 0


def calculate_iou(pred, label):
    intersection = ((label == 1) & (pred == 1)).sum()
    union = ((label == 1) | (pred == 1)).sum()
    if not union:
        return 0
    else:
        iou = intersection.item() / union.item()
        return iou
