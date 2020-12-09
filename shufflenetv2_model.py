import torch
import torch.nn as nn

def split(x, groups):
	out = x.chunk(groups, dim=1)
	return out

def shuffle(x, groups):
	N, C, H, W = x.size()
	return x.view(N, groups, int(C/groups), H, W).permute(0, 2, 1, 3, 4).contiguous().view(N, C, H, W)

class ShuffleUnit(nn.Module):
	def __init__(self, in_channels, out_channels, stride):
		super().__init__()

		mid_channels = out_channels // 2
		if stride > 1:
			self.branch1 = nn.Sequential(
					nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels, bias=False),
					nn.BatchNorm2d(in_channels),
					nn.Conv2d(in_channels, mid_channels, 1, bias=False),
					nn.BatchNorm2d(mid_channels),
					nn.ReLU(inplace=True))

			self.branch2 = nn.Sequential(
					nn.Conv2d(in_channels, mid_channels, 1, bias=False),
					nn.BatchNorm2d(mid_channels),
					nn.ReLU(inplace=True),
					nn.Conv2d(mid_channels, mid_channels, 3, stride=stride, padding=1, groups=mid_channels, bias=False),
					nn.BatchNorm2d(mid_channels),
					nn.Conv2d(mid_channels, mid_channels, 1, bias=False),
					nn.BatchNorm2d(mid_channels),
					nn.ReLU(inplace=True))
		else:
			self.branch1 = nn.Sequential()
			self.branch2 = nn.Sequential(
					nn.Conv2d(mid_channels, mid_channels, 1, bias=False),
					nn.BatchNorm2d(mid_channels),
					nn.ReLU(inplace=True),
					nn.Conv2d(mid_channels, mid_channels, 3, stride=stride, padding=1, groups=mid_channels, bias=False),
					nn.BatchNorm2d(mid_channels),
					nn.Conv2d(mid_channels, mid_channels, 1, bias=False),
					nn.BatchNorm2d(mid_channels),
					nn.ReLU(inplace=True))

		self.stride = stride

	def forward(self, x):
		if self.stride == 1:
			x1, x2 = split(x, 2)
			out = torch.cat((self.branch1(x1), self.branch2(x2)), dim=1)
		else:
			out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
		out = shuffle(out, 2)
		return out

class ShuffleNetV2(nn.Module):
	def __init__(self, channel_num, class_num=2):
		super().__init__()

		self.conv1 = nn.Sequential(
				nn.Conv2d(3, 24, 3, stride=2, padding=1, bias=False),
				nn.BatchNorm2d(24),
				nn.ReLU(inplace=True))

		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

		self.stage2 = self.make_layers(24, channel_num[0], 4, 2)
		self.stage3 = self.make_layers(channel_num[0], channel_num[1], 8, 2)
		self.stage4 = self.make_layers(channel_num[1], channel_num[2], 4, 2)

		self.conv5 = nn.Sequential(
				nn.Conv2d(channel_num[2], 1024, 1, bias=False),
				nn.BatchNorm2d(1024),
				nn.ReLU(inplace=True))

		self.avgpool = nn.AdaptiveAvgPool2d(1)
		self.fc = nn.Linear(1024, class_num)

	def make_layers(self, in_channels, out_channels, repeat, stride):
		layers = []
		layers.append(ShuffleUnit(in_channels, out_channels, stride))
		in_channels = out_channels
		
		for i in range(repeat - 1):
			layers.append(ShuffleUnit(in_channels, out_channels, 1))

		return nn.Sequential(*layers)

	def forward(self, x):
		out = self.conv1(x)
		out = self.maxpool(out)
		out = self.stage2(out)
		out = self.stage3(out)
		out = self.stage4(out)
		out = self.conv5(out)
		out = self.avgpool(out)
		out = out.flatten(1)
		out = self.fc(out)

		return out 

channel_num = [48, 96, 192]
model = ShuffleNetV2(channel_num, class_num=2)