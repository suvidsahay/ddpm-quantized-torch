import torch
import torch.nn as nn

class QuantizeWeights(nn.Module):
    def __init__(self, bit_width):
        super(QuantizeWeights, self).__init__()
        self.bit_width = bit_width
        self.interval = nn.Parameter(torch.tensor(1.0), requires_grad=False)

    def initialize_interval(self, w):
        max_val = w.max().item()
        min_val = w.min().item()
        self.interval.data = (max_val - min_val) / (2 ** (self.bit_width - 1) - 1)

    def forward(self, w):
        max_val = 2 ** (self.bit_width - 1) - 1
        min_val = -2 ** (self.bit_width - 1)
        w_clamped = torch.clamp(w / self.interval, min_val, max_val)
        w_quantized = torch.round(w_clamped) * self.interval
        return w_quantized

    def backward(self, grad_output):
        return grad_output  # STE: gradient approximation

class QuantizeActivations(nn.Module):
    def __init__(self, bit_width):
        super(QuantizeActivations, self).__init__()
        self.bit_width = bit_width
        self.interval = nn.Parameter(torch.tensor(1.0), requires_grad=False)

    def initialize_interval(self, x):
        max_val = x.max().item()
        self.interval.data = max_val / (2 ** self.bit_width - 1)

    def forward(self, x):
        max_val = 2 ** self.bit_width - 1
        x_clamped = torch.clamp(x / self.interval, 0, max_val)
        x_quantized = torch.round(x_clamped) * self.interval
        return x_quantized

    def backward(self, grad_output):
        return grad_output  # STE: gradient approximation