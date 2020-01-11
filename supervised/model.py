from torch import nn


class BasicBlock(nn.Module):
    def __init__(self, in_channels, channels, bias=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels,
                               channels,
                               3,
                               stride=1,
                               padding=1,
                               bias=bias)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, padding=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu1(y)
        y = self.conv2(y)
        y = self.bn2(y)
        x = x + y
        x = self.relu2(x)
        return x


class ResNet(nn.Module):
    def __init__(self, in_channels, layers, channels, bias=False):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels,
                      channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=bias),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
        self.convs = nn.ModuleList(
            [BasicBlock(channels, channels, bias) for _ in range(layers)])

    def forward(self, x):
        x = self.conv1(x)
        for conv in self.convs:
            x = conv(x)
        return x


class AlphaZero(nn.Module):
    def __init__(self, in_channels, layers=10, channels=128, bias=False):
        super().__init__()
        self.resnet = ResNet(in_channels, layers, channels, bias)
        # policy head
        self.policy_head_front = nn.Sequential(
            nn.Conv2d(channels, 2, 1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
        )
        self.policy_head_end = nn.Linear(2 * 81, 81)
        # value head
        self.value_head_front = nn.Sequential(
            nn.Conv2d(channels, 1, 1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
        )
        self.value_head_end = nn.Sequential(
            nn.Linear(81, channels),
            nn.ReLU(),
            nn.Linear(channels, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.resnet(x)
        # policy head
        p = self.policy_head_front(x)
        p = p.view(-1, 2 * 81)
        p = self.policy_head_end(p)
        # value head
        v = self.value_head_front(x)
        v = v.view(-1, 81)
        v = self.value_head_end(v)
        return p, v
