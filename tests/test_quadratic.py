import pytest
import torch

from lr_adaptation import QuadraticAdaptation
from tests.conftest import needs_cuda

# def scale_learning_rate(self, scaling=1.0):
#     for group in self.optimizer.param_groups:
#         group["lr"] = group["lr"] * scaling


@pytest.mark.parametrize("device_type", ["cpu", pytest.param("cuda", marks=needs_cuda)], ids=["cpu", "cuda"])
def test_quadratic_increase(device_type: str, seed: int = 42):
    torch.manual_seed(seed)
    print("Executing standalone")
    device = torch.device(device_type)
    # device = torch.device("cpu")
    print(device)
    N = 64
    M = 3
    M1 = 32
    O = 1

    lr = 0.01
    X = 0.3 * torch.randn(N, M).to(device)
    Y = (5 + torch.randn(N, O)).to(device)
    net = torch.nn.Sequential(torch.nn.Linear(M, M1), torch.nn.ReLU(), torch.nn.Linear(M1, O))
    net.to(device)

    # nn.CrossEntropyLoss(reduction='none')
    criterion = torch.nn.MSELoss(reduction="none")

    # optimizer = optim.SGD(net.parameters(), lr=lr)
    # optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.1)
    # optimizer = optim.Adagrad(net.parameters(), lr=lr)
    # optimizer = optim.RMSprop(net.parameters(), lr=lr)
    optimizer = torch.optim.Adam(net.parameters(), betas=(0.2, 0.9), lr=lr)

    lr_optimizer = QuadraticAdaptation(optimizer)
    # print(lr_optimizer.optimizer.)

    def closure():
        lr_optimizer.zero_grad()
        outputs = net(X)
        batch_loss = criterion(outputs, Y)
        loss = torch.mean(batch_loss)
        return loss

    lr_optimizer.step(closure)
    assert lr_optimizer.state["accumulated_scaling"] > 1
