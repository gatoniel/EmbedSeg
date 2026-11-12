from EmbedSeg.criterions.my_loss import SpatialEmbLoss
from EmbedSeg.criterions.my_loss_3d import SpatialEmbLoss_3d
from EmbedSeg.criterions.my_loss_tracking import SpatialEmbTrackingLoss


def get_loss(
    grid_z,
    grid_y,
    grid_x,
    pixel_z,
    pixel_y,
    pixel_x,
    one_hot,
    loss_opts,
):
    if loss_opts.get("tracking", False):
        # opt-in: segmentation + tracking loss (see my_loss_tracking.py)
        opts = {k: v for k, v in loss_opts.items() if k != "tracking"}
        return SpatialEmbTrackingLoss(
            grid_z, grid_y, grid_x, pixel_z, pixel_y, pixel_x, one_hot, **opts
        )
    if grid_z is None:
        return SpatialEmbLoss(grid_y, grid_x, pixel_y, pixel_x, one_hot, **loss_opts)
    else:
        return SpatialEmbLoss_3d(
            grid_z, grid_y, grid_x, pixel_z, pixel_y, pixel_x, one_hot, **loss_opts
        )
