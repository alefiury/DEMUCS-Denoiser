import math

import torch
import julius
from torch import nn
from torch.nn import functional as F

from utils.utils import center_trim


class DownConvBlock(nn.Module):
    """
    Convolutional block followed by a GLU activation function.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
    ):
        super(DownConvBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        self.relu_activation = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, 2*out_channels, kernel_size=1, stride=1)
        self.glu_activation = nn.GLU(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.relu_activation(x)
        x = self.conv2(x)
        return self.glu_activation(x)


class UpConvlock(nn.Module):
    """
    Transposed convolutional block followed by a GLU activation function.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        is_last_layer: bool = False
    ):
        super(UpConvlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, 2*in_channels, kernel_size=1, stride=1)
        self.glu_activation = nn.GLU(dim=1)
        self.up_conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        if not is_last_layer:
            self.relu_activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.glu_activation(x)
        return self.up_conv(x) if hasattr(self, 'relu_activation') else self.up_conv(x)


class DemucsDenoiser(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        hidden_channels: int = 64,
        conv_stride: int = 2,
        conv_kernel_size: int = 8,
        num_layers: int = 5,
        use_bidirectional: bool = True,
        upsampling_rate: int = 2,
        normalize_input: bool = True,
        apply_normalization: bool = True,
        stability_constant: float = 1e-5,
        training_sample_rate: int = 44100,
    ):

        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.conv_kernel_size = conv_kernel_size
        self.conv_stride = conv_stride
        self.use_bidirectional = use_bidirectional
        self.stability_constant = stability_constant
        self.upsampling_rate = upsampling_rate
        self.normalize_input = normalize_input
        self.apply_normalization = apply_normalization
        self.training_sample_rate = training_sample_rate

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()

        self.bilstm = nn.LSTM(
            input_size=hidden_channels * (2 ** (num_layers - 1)),
            hidden_size=hidden_channels * (2 ** (num_layers - 1)),
            num_layers=2,
            bidirectional=use_bidirectional
        )

        self.proj = nn.Linear(
            2 * hidden_channels * (2 ** (num_layers - 1)),
            hidden_channels * (2 ** (num_layers - 1))
        )

        # Encoder layers
        for i in range(num_layers):
            in_ch = 1 if i == 0 else hidden_channels * (2 ** (i - 1))
            out_ch = hidden_channels * (2 ** i)
            self.encoders.append(DownConvBlock(in_ch, out_ch, conv_kernel_size, conv_stride))

        # Decoder layers
        for i in range(num_layers):
            in_ch = hidden_channels * (2 ** (num_layers - i - 1))
            out_ch = hidden_channels * (2 ** (num_layers - i - 2)) if i < num_layers - 1 else 1
            self.decoders.append(UpConvlock(in_ch, out_ch, conv_kernel_size, conv_stride, is_last_layer=True if i == num_layers - 1 else False))

    def valid_length(self, length: int) -> int:
        """
        Return the nearest valid length to use with the model so that
        there is no time steps left over in a convolution, e.g. for all
        layers, size of the input - kernel_size % stride = 0.

        Note that input are automatically padded if necessary to ensure that the output
        has the same length as the input.

        Args:

            length (int): The length of the input signal.

        Returns:

            int: The nearest valid length to use with the model.
        """
        length *= self.upsampling_rate

        for _ in range(self.num_layers):
            length = math.ceil((length - self.conv_kernel_size) / self.conv_stride) + 1
            length = max(1, length)

        for _ in range(self.num_layers):
            length = (length - 1) * self.conv_stride + self.conv_kernel_size

        length = math.ceil(length / self.upsampling_rate)
        return int(length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_length = x.size(-1)
        skips = []

        if self.normalize_input:
            mono = x.mean(dim=1, keepdim=True)
            mean = mono.mean(dim=-1, keepdim=True)
            std = mono.std(dim=-1, keepdim=True)
            x = (x - mean) / (self.stability_constant + std)
        else:
            mean = 0
            std = 1

        delta = self.valid_length(original_length) - original_length
        x = F.pad(x, (0, delta))

        # Upsample
        x = julius.resample_frac(x, 1, self.upsampling_rate)

        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)

        x, _ = self.bilstm(x.permute(2, 0, 1))
        x = self.proj(x)
        x = x.permute(1, 2, 0)

        for i, decoder in enumerate(self.decoders):
            skip = skips[-i-1]
            # Calculate the size difference in the last dimension between 'x' and 'skip'
            size_diff = skip.size(-1) - x.size(-1)
            # Trim 'skip' to match the size of 'x' in the last dimension
            trimmed_skip = skip[..., :skip.size(-1) - size_diff]
            # Add the trimmed skip connection to 'x'
            x += trimmed_skip
            x = decoder(x)

        # Downsample
        x = julius.resample_frac(x, self.upsampling_rate, 1)

        # Reshape output to match the original input shape
        x = x * std + mean
        x = center_trim(x, original_length)
        x = x.view(x.size(0), 1, x.size(-1))
        return x


def test():
    import argparse
    parser = argparse.ArgumentParser(
        "denoiser.demucs",
        description="Benchmark the streaming Demucs implementation, "
                    "as well as checking the delta with the offline implementation."
    )
    parser.add_argument("--depth", default=5, type=int)
    parser.add_argument("--resample", default=4, type=int)
    parser.add_argument("--hidden", default=48, type=int)
    parser.add_argument("--sample-rate", default=48000, type=float)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("-t", "--num_threads", type=int)
    args = parser.parse_args()
    if args.num_threads:
        torch.set_num_threads(args.num_threads)
    sr = args.sample_rate
    demucs = DemucsDenoiser().to(args.device)

    # print(demucs)
    x = torch.randn(1, int(sr * 4)).to(args.device)
    print(x.shape)
    # print(x[None].shape)
    out = demucs(x[None])[0]
    assert out.shape == x.shape, f"{out.shape} != {x.shape}"
    print("All good!")
    model_size = sum(p.numel() for p in demucs.parameters()) * 4 / 2**20
    print(f"model size: {model_size:.1f}MB, ", end='')


if __name__ == "__main__":
    test()