import torch
import torch.nn as nn
import EmbedSeg.models.erfnet_3d as erfnet


class BranchedERFNet_3d(nn.Module):
    def __init__(self, num_classes, input_channels=1, encoder=None):
        super().__init__()

        print("Creating Branched Erfnet 3D with {} outputs".format(num_classes))
        if encoder is None:
            self.encoder = erfnet.Encoder(sum(num_classes), input_channels)
        else:
            self.encoder = encoder

        self.decoders = nn.ModuleList()
        for n in num_classes:
            self.decoders.append(erfnet.Decoder(n))

    # The first two decoders (offsets + sigma, seed) form the segmentation head.
    # Any further decoder (e.g. tracking: offsets + sigma + division flag) is
    # optional and skipped when forward(..., heads="seg") is used.
    n_seg_decoders = 2

    def init_output(self, n_sigma=1):
        with torch.no_grad():
            # initialise the segmentation head and, if present, the tracking head
            for idx in [0, 2]:
                if idx >= len(self.decoders):
                    continue
                output_conv = self.decoders[idx].output_conv
                print("initialize last layer with size: ", output_conv.weight.size())

                output_conv.weight[:, 0:3, :, :, :].fill_(0)
                output_conv.bias[0:3].fill_(0)

                output_conv.weight[:, 3 : 3 + n_sigma, :, :, :].fill_(0)
                output_conv.bias[3 : 3 + n_sigma].fill_(1)

    def forward(self, input, only_encode=False, heads="all"):
        """heads="all" runs every decoder; heads="seg" skips the tracking
        decoder(s) and returns only the segmentation channels."""
        if only_encode:
            return self.encoder.forward(input, predict=True)
        else:
            output = self.encoder(input)

        decoders = self.decoders
        if heads == "seg":
            decoders = self.decoders[: self.n_seg_decoders]
        elif heads != "all":
            raise ValueError("heads must be 'all' or 'seg', got {!r}".format(heads))
        return torch.cat([decoder.forward(output) for decoder in decoders], 1)
