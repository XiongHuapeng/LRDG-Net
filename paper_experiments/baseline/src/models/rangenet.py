import torch
import torch.nn as nn


class ConvBNLeaky(nn.Sequential):
  def __init__(self, cin, cout, kernel_size, stride=1, padding=0):
    super().__init__(
      nn.Conv2d(cin, cout, kernel_size, stride=stride, padding=padding, bias=False),
      nn.BatchNorm2d(cout),
      nn.LeakyReLU(0.1, inplace=True),
    )


class DarkResidual(nn.Module):
  """Darknet-style 1x1 -> 3x3 residual unit."""
  def __init__(self, channels):
    super().__init__()
    hidden = max(channels // 2, 1)
    self.body = nn.Sequential(
      ConvBNLeaky(channels, hidden, 1),
      ConvBNLeaky(hidden, channels, 3, padding=1),
    )

  def forward(self, x):
    return x + self.body(x)


class RangeStage(nn.Module):
  """RangeNet-style stage: preserve vertical resolution, halve horizontal width."""
  def __init__(self, cin, cout, num_residual):
    super().__init__()
    layers = [ConvBNLeaky(cin, cout, 3, stride=(1, 2), padding=1)]
    layers += [DarkResidual(cout) for _ in range(int(num_residual))]
    self.net = nn.Sequential(*layers)

  def forward(self, x):
    return self.net(x)


class RangeNetClassifier(nn.Module):
  """RangeNet-style range-view encoder for frame classification.

  Input: [B, 5, H, W]
   0: range / 80
   1: x / 80
   2: y / 80
   3: z / 80
   4: intensity / 255

  Design principle:
   - preserve the RangeNet/Darknet-style depth and horizontal-downsampling pattern;
   - retain global average pooling and a lightweight binary head.

  Channel schedule:
   16 -> 32 -> 64 -> 128 -> 256 -> 512

  Residual schedule:
   [1, 1, 2, 2, 1]

    """

  def __init__(self, num_classes=2, dropout=0.3):
    super().__init__()

    self.stem = ConvBNLeaky(5, 16, 3, padding=1)

    # Same stage/residual depth as the previous adapted baseline:
    # stem + 5 downsample convs + 2*(1+1+2+2+1) residual convs = 20 conv layers.
    # Width is reduced, not depth.
    self.stage1 = RangeStage(16, 32, 1)
    self.stage2 = RangeStage(32, 64, 1)
    self.stage3 = RangeStage(64, 128, 2)
    self.stage4 = RangeStage(128, 256, 2)
    self.stage5 = RangeStage(256, 512, 1)

    self.pool = nn.AdaptiveAvgPool2d((1, 1))
    self.classifier = nn.Sequential(
      nn.Flatten(),
      nn.Linear(512, 128),
      nn.LeakyReLU(0.1, inplace=True),
      nn.Dropout(dropout),
      nn.Linear(128, num_classes),
    )

  def forward(self, x):
    x = self.stem(x)
    x = self.stage1(x)
    x = self.stage2(x)
    x = self.stage3(x)
    x = self.stage4(x)
    x = self.stage5(x)
    return self.classifier(self.pool(x))
