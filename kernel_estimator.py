import numpy as np
import torch


class EpanechnikovKernel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return 3/4 * (1 - torch.pow(input.clamp(min=-1, max=1), 2))
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return -2 * 3/4 * input.clamp(min=-1, max=1) * (1 - torch.pow(input.clamp(min=-1, max=1), 2)) * grad_output


class UniformKernel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        fltr = torch.abs(input) > 1
        output = 1 / 2 * torch.ones_like(input)
        output[fltr] = 0
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return torch.zeros_like(input)


class GaussianKernel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return 1 / torch.sqrt(2 * torch.pi) * torch.exp(-torch.pow(input, 2) / 2)
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return -input * 1 / torch.sqrt(2 * torch.pi) * torch.exp(-torch.pow(input, 2) / 2) * grad_output


def KernelEstimator(input, density, h):
    """
    input indicates the input variable (scalar)
    density indicates the observed probability density (vector)
    h indicates the scale factor of kernel estimator (scalar)
    """

    return EpanechnikovKernel.apply((input.view(-1, 1).repeat((1, density.size(0))) - density) / h).mean(1) / h

