from torch import nn


class conv3x3_to_outfun(nn.Module):
    def __init__(self, in_planes, out_planes, relu_or_drop, stride=(1, 1), padding=(1, 1)):
        super(conv3x3_to_outfun, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=(3, 3), stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        if relu_or_drop == 'relu':
            self.last = nn.ReLU(inplace=True)
        elif relu_or_drop == 'drop':
            self.last = nn.Dropout(inplace=True, p=0.5)
        else:
            self.last = nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.last(x)
        return x