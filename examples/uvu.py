"""Uvu with single-channel angular input and radial-network path weights."""
import torch
from ictc_o2_hp import UvuTensorProduct


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    tp = UvuTensorProduct(
        128, 1, 6, parity="full", parity_in2="canonical",
        shared_weights=False,
    ).cuda().float()
    radial_network = torch.nn.Linear(16, tp.weight_numel).cuda().float()
    rows = 256
    left = torch.randn(rows, tp.input1_dimension, device="cuda", requires_grad=True)
    angular = torch.randn(rows, tp.input2_dimension, device="cuda", requires_grad=True)
    radial = torch.randn(rows, 16, device="cuda", requires_grad=True)
    weights = radial_network(radial)
    compiled = torch.compile(tp, fullgraph=True, dynamic=False)
    output = compiled(left, angular, weights)
    output.square().mean().backward()
    assert left.grad is not None and angular.grad is not None and radial.grad is not None
    assert radial_network.weight.grad is not None
    print(output.shape)


if __name__ == "__main__":
    main()
