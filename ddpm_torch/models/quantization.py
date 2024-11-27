import torch
import torch.nn as nn

class QuantizeWeights(nn.Module):
    def __init__(self, bit_width):
        super(QuantizeWeights, self).__init__()
        self.bit_width = bit_width
        self.interval = nn.Parameter(torch.tensor(1.0), requires_grad=True)  # IW

    def initialize_interval(self, w):
        with torch.no_grad():
            max_val = w.max().item()
            min_val = w.min().item()
            # Assign a tensor instead of a float
            self.interval.data = torch.tensor(
                (max_val - min_val) / (2 ** (self.bit_width - 1) - 1),
                device=self.interval.device,  # Ensure compatibility with the device
                dtype=self.interval.dtype     # Ensure data type matches
            )

    def forward(self, w):
        # Clamp the interval to prevent invalid values
        self.interval.data.clamp_(1e-5, float("inf"))

        # Define quantization range for weights
        max_val = 2 ** (self.bit_width - 1) - 1
        min_val = -2 ** (self.bit_width - 1)

        # Clamping
        w_clamped = torch.clamp(w / self.interval, min_val, max_val)

        # Quantization (with STE for backprop)
        w_quantized = torch.round(w_clamped) * self.interval

        # Apply straight-through estimator (STE) for backpropagation
        return w_quantized


class QuantizeActivations(nn.Module):
    def __init__(self, bit_width):
        super(QuantizeActivations, self).__init__()
        self.bit_width = bit_width
        self.interval = nn.Parameter(torch.tensor(1.0), requires_grad=True)  # IX

    def initialize_interval(self, x):
        with torch.no_grad():
            max_val = x.max().item()
            # Assign a tensor instead of a float
            self.interval.data = torch.tensor(
                max_val / (2 ** self.bit_width - 1),
                device=self.interval.device,  # Ensure compatibility with the device
                dtype=self.interval.dtype     # Ensure data type matches
            )

    def forward(self, x):
        # Clamp the interval to prevent invalid values
        self.interval.data.clamp_(1e-5, float("inf"))

        # Define quantization range for activations
        max_val = 2 ** self.bit_width - 1

        # Clamping
        x_clamped = torch.clamp(x / self.interval, 0, max_val)

        # Quantization (with STE for backprop)
        x_quantized = torch.round(x_clamped) * self.interval

        # Apply straight-through estimator (STE) for backpropagation
        return x_quantized
