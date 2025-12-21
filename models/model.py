import torch
import torch.nn as nn
from torchvision import models


class DepthwiseSeparable(nn.Module):
    def __init__(self, in_channels, out_channels, expansion=3):
        super(DepthwiseSeparable, self).__init__()

        hidden_channels = in_channels * expansion

        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1, groups=hidden_channels, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        tmp = x
        out = self.conv(x)
        tmp = self.downsample(tmp)
        out += tmp
        return out

class Block(nn.Module):
    def __init__(self, in_channels, growth_rate=16):
        super(Block, self).__init__()

        inter_channels = 4 * growth_rate

        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=inter_channels, kernel_size=1, stride=1, padding=0, bias=False)
        )

        self.depthwise_separable = nn.Sequential(
            DepthwiseSeparable(in_channels=inter_channels, out_channels=growth_rate),
            nn.BatchNorm2d(growth_rate),
            nn.ReLU(),
        )

    def forward(self, x):
        tmp = x

        out = self.conv1(x)

        out = self.depthwise_separable(out)

        return torch.cat([tmp, out], dim=1)

class Transition(nn.Module):
    def __init__(self, in_channels):
        super(Transition, self).__init__()

        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        )

    def forward(self, x):
        return self.conv(x)

class TeacherBranch(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes):
        super(TeacherBranch, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=2, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.avgpool = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(output_size=(1, 1))
        )

        self.fc = nn.Linear(in_features=out_channels, out_features=num_classes, bias=True)

    def forward(self, x):
        tmp = x
        out = self.conv(x)

        tmp = self.downsample(tmp)
        out += tmp

        out = self.avgpool(out)
        fea = out.view(out.size(0), -1)
        out = self.fc(fea)

        return fea, out

class StudentBranch(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(StudentBranch, self).__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_channels),
        )

        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_channels),
        )

        self.avgpool = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(output_size=(1, 1))
        )

        self.fc = nn.Linear(in_features=in_channels, out_features=num_classes, bias=True)

    def forward(self, x):
        tmp = x
        out = self.conv(x)

        tmp = self.downsample(tmp)
        out += tmp

        out = self.avgpool(out)
        fea = out.view(out.size(0), -1)
        out = self.fc(fea)

        return fea, out

class Student(nn.Module):
    def __init__(self, num_classes, num_layers, growth_rate=16):
        super(Student, self).__init__()

        self.in_channels = 64

        self.growth_rate = growth_rate

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.layer1 = self._make_layer(num_layers[0])
        self.transition1 = Transition(self.in_channels)

        self.layer2 = self._make_layer(num_layers[1])
        self.transition2 = Transition(self.in_channels)

        self.layer3 = self._make_layer(num_layers[2])
        self.transition3 = Transition(self.in_channels)

        self.layer4 = self._make_layer(num_layers[3])

        self.avgpool = nn.Sequential(
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.fc = nn.Linear(self.in_channels, num_classes, bias=True)

        self.branch = StudentBranch(352, num_classes)

    def _make_layer(self, number_block):
        layers = []

        for i in range(number_block):

            layers.append(Block(in_channels=self.in_channels, growth_rate=self.growth_rate))
            self.in_channels += self.growth_rate

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.stem(x)

        out = self.layer1(out)
        out = self.transition1(out) # [1, 128, 28, 28]

        out = self.layer2(out)
        out = self.transition2(out) # [1, 224, 14, 14]

        out = self.layer3(out)
        out = self.transition3(out) # [1, 352, 7, 7]

        fea1, logit1 = self.branch(out)

        out = self.layer4(out) # [1, 512, 7, 7]
        out = self.avgpool(out)
        fea2 = out.view(out.size(0), -1)

        logit2 = self.fc(fea2)

        return fea1, logit1, fea2, logit2


class Teacher(nn.Module):
    def __init__(self, num_classes):
        super(Teacher, self).__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        self.stem = nn.Sequential(resnet.conv1,
                                  resnet.bn1,
                                  resnet.relu,
                                  resnet.maxpool)

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.fc = nn.Linear(resnet.fc.in_features, num_classes)

        self.branch = TeacherBranch(in_channels=256, out_channels=352, num_classes=num_classes)

    def forward(self, x):
        out = self.stem(x)

        out = self.layer1(out)

        out = self.layer2(out)

        out = self.layer3(out)

        fea1, logit1 = self.branch(out)

        out = self.layer4(out)

        out = self.avgpool(out)

        fea2 = out.view(out.size(0), -1)

        logit2 = self.fc(fea2)

        return fea1, logit1, fea2, logit2