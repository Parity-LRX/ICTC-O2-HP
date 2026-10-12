"""Run this example only inside an authorized remote batch job."""
import torch
from ictc_o2_hp import TensorProduct


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    module = TensorProduct(64, 4, parity="full").cuda().float()
    x, y = (torch.randn(6400, module.representation_dimension, device="cuda", dtype=torch.float32,
                        requires_grad=True) for _ in range(2))
    operation = torch.compile(module, fullgraph=True, dynamic=False)
    output = operation(x, y)
    gradients = torch.autograd.grad(output, (x, y, module.weight), torch.ones_like(output))
    print(output.shape, tuple(g.shape for g in gradients))


if __name__ == "__main__":
    main()
