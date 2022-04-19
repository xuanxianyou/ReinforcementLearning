from torch import nn


class MLP(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.mlp = nn.Sequential(
            nn.Linear(self.num_inputs, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, self.num_outputs)
        )

    def forward(self, x):
        return self.mlp(x)