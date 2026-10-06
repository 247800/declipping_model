import torch
from torch import nn

def center_crop_1d(x, target_length):
    """
    Applying skip connections requires cropping feature maps from the contracting path
    to match the size of the corresponding feature maps in the expanding path.
    (The size mismatch is caused by no padding in U-net's convolutions)
    """
    current_length = x.size(-1)
    if current_length < target_length:
        raise ValueError(
            f"Cannot crop tensor from length {current_length} "
            f"to larger target length {target_length}."
        )
    difference = current_length - target_length
    left = difference // 2
    right = left + target_length
    return x[:, :, left:right]

class DoubleConv1D(nn.Module):
    """
    One contracting-path block of the U-Net
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.layers(x)

class UpBlock1D(nn.Module):
    """
    One expansive-path block of the U-Net
    """
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upconv = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=2,
            stride=2,
            padding=0,
            bias=False,
        )
        self.double_conv = DoubleConv1D(
            in_channels=out_channels + skip_channels,
            out_channels=out_channels,
        )
    def forward(self, x, skip_features):
        x = self.upconv(x)
        skip_features = center_crop_1d(x=skip_features, target_length=x.size(-1))
        x = torch.cat((skip_features, x), dim=1)
        return self.double_conv(x)

class DeclippingUNet(nn.Module):
    """
    Bias-free U-Net model for audio declipping with 5 down + 5 up blocks
    (following https://arxiv.org/pdf/2602.22279 section 5.3)
    """
    def __init__(self, input_channels=1, output_channels=1):
        super().__init__()
        if input_channels != 1:
            raise ValueError(
                "This waveform version currently expects input_channels=1. "
                "For stereo, either train one model per channel or remove "
                "this validation and set input_channels=2."
            )

        # Contracting path
        self.enc1 = DoubleConv1D(input_channels, 64)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.enc2 = DoubleConv1D(64, 128)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.enc3 = DoubleConv1D(128, 256)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.enc4 = DoubleConv1D(256, 512)
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.enc5 = DoubleConv1D(512, 1024)
        self.pool5 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Bottom of the U
        self.bottleneck = DoubleConv1D(1024, 2048)

        # Expansive path with copy-and-crop skip connections
        self.dec5 = UpBlock1D(
            in_channels=2048,
            skip_channels=1024,
            out_channels=1024,
        )
        self.dec4 = UpBlock1D(
            in_channels=1024,
            skip_channels=512,
            out_channels=512,
        )
        self.dec3 = UpBlock1D(
            in_channels=512,
            skip_channels=256,
            out_channels=256,
        )
        self.dec2 = UpBlock1D(
            in_channels=256,
            skip_channels=128,
            out_channels=128,
        )
        self.dec1 = UpBlock1D(
            in_channels=128,
            skip_channels=64,
            out_channels=64,
        )

        self.output_conv = nn.Conv1d(
            in_channels=64,
            out_channels=output_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

    def forward(self, x):
        input_was_2d = x.ndim == 2
        if input_was_2d:
            # [batch, samples] -> [batch, 1, samples]
            x = x.unsqueeze(1)
        if x.ndim != 3:
            raise ValueError(
                "Input must have shape [batch, samples] "
                "or [batch, channels, samples]."
            )
        if x.size(1) != 1:
            raise ValueError(
                f"Expected one waveform channel, received {x.size(1)}."
            )

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        e5 = self.enc5(self.pool4(e4))

        # Bottleneck
        b = self.bottleneck(self.pool5(e5))

        # Decoder
        d5 = self.dec5(b, e5)
        d4 = self.dec4(d5, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)

        y = self.output_conv(d1)

        if input_was_2d and y.size(1) == 1:
            y = y.squeeze(1)

        return y