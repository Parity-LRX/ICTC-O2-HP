"""Radial MLP / angular V=1 TP, including first gradients through geometry.

For force-loss double backward on PyTorch 2.7.1, call the ordinary outer
model instead of torch.compile(model). Internal bounded blocks still compile.
"""

import torch

from ictc_o2_hp import UvuTensorProduct, RadialUvuTensorProduct


class RadialChain(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = torch.nn.Sequential(
            torch.nn.Linear(16, 64), torch.nn.SiLU(),
            torch.nn.Linear(64, 64), torch.nn.SiLU(),
        )
        tp = UvuTensorProduct(128, 1, 6, parity="full", parity_in2="canonical",
                              shared_weights=False)
        self.product = RadialUvuTensorProduct(
            tp, torch.nn.Linear(64, tp.weight_numel, bias=False)
        )

    def forward(self, features, angular, radial_basis):
        return self.product(features, angular, self.trunk(radial_basis))


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    model = RadialChain().cuda().float()
    features = torch.randn(64, model.product.tp.input1_dimension,
                            device="cuda", requires_grad=True)
    radial_basis = torch.randn(64, 16, device="cuda", requires_grad=True)
    theta = torch.randn(64, 1, device="cuda", requires_grad=True)
    angular = torch.cat([torch.ones_like(theta)] + [
        component for m in range(1, 7)
        for component in (torch.cos(m * theta), torch.sin(m * theta))
    ], dim=-1)
    output = torch.compile(model, fullgraph=True, dynamic=False)(
        features, angular, radial_basis
    )
    output.square().mean().backward()
    assert features.grad is not None and radial_basis.grad is not None and theta.grad is not None
    assert all(parameter.grad is not None for parameter in model.parameters())
    print(output.shape)


if __name__ == "__main__":
    main()
