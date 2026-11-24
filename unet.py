import torch
from torch import nn
import math

def center_crop_1d(x, target_length):
    """Crop the last dimension while preserving leading dimensions."""
    current_length = x.size(-1)
    if current_length < target_length:
        raise ValueError(f"Cannot crop length {current_length} to larger length {target_length}.")
    left = (current_length - target_length) // 2
    return x[..., left:left + target_length]

def prepare_unet_input(x):
    """Crop a waveform batch to a valid input length for U-Net."""
    current_length = x.size(-1)
    if current_length < 380:
        raise ValueError(f"U-Net requires at least 380 input samples, received {current_length}.")
    target_length = current_length - ((current_length - 28) % 32)
    return center_crop_1d(x, target_length)

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
        crop_difference = skip_features.size(-1) - x.size(-1)
        if crop_difference < 0 or crop_difference % 2 != 0:
            raise ValueError(
                "Skip connection requires a non-negative, even "
                f"length difference; got {crop_difference}. "
                f"Skip length={skip_features.size(-1)}, "
                f"upsampled length={x.size(-1)}."
            )
        skip_features = center_crop_1d(
            x=skip_features,
            target_length=x.size(-1),
        )
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
        self.enc1 = DoubleConv1D(input_channels, 16)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.enc2 = DoubleConv1D(16, 32)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.enc3 = DoubleConv1D(32, 64)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.enc4 = DoubleConv1D(64, 128)
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.enc5 = DoubleConv1D(128, 256)
        self.pool5 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Bottom of the U
        self.bottleneck = DoubleConv1D(256, 512)

        # Expansive path with copy-and-crop skip connections
        self.dec5 = UpBlock1D(
            in_channels=512,
            skip_channels=256,
            out_channels=256,
        )
        self.dec4 = UpBlock1D(
            in_channels=256,
            skip_channels=128,
            out_channels=128,
        )
        self.dec3 = UpBlock1D(
            in_channels=128,
            skip_channels=64,
            out_channels=64,
        )
        self.dec2 = UpBlock1D(
            in_channels=64,
            skip_channels=32,
            out_channels=32,
        )
        self.dec1 = UpBlock1D(
            in_channels=32,
            skip_channels=16,
            out_channels=16,
        )

        self.output_conv = nn.Conv1d(
            in_channels=16,
            out_channels=output_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

        self.initialize_weights()

    def initialize_weights(self):
        """Initialize weights for the current bias-free architecture."""
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nonlinearity = (
                    "linear" if module is self.output_conv else "relu"
                )
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_in",
                    nonlinearity=nonlinearity,
                )

            elif isinstance(module, nn.ConvTranspose1d):
                # This rule assumes non-overlapping upsampling.
                if (
                    module.kernel_size != (2,)
                    or module.stride != (2,)
                    or module.padding != (0,)
                    or module.output_padding != (0,)
                    or module.dilation != (1,)
                    or module.groups != 1
                ):
                    raise ValueError(
                        "Recheck initialization for the modified upconvolution."
                    )

                # Each output sample sums one value per input channel.
                # No activation immediately follows this upconvolution.
                nn.init.normal_(
                    module.weight,
                    mean=0.0,
                    std=1.0 / math.sqrt(module.in_channels),
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

        # verify signal length to ensure symmetrical crop
        length = x.size(-1)
        if length < 380 or (length - 28) % 32 != 0:
            raise ValueError(
                f"Invalid U-Net input length: {length}. "
                "Expected length >= 380 and "
                "(length - 28) % 32 == 0. "
                "Call prepare_unet_input before model(...)."
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