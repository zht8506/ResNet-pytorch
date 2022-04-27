import torch
from torch import nn

# 因为resnet的18、34层，和50层以上有不同的基础结构，所以要设置两种基本残差卷积

# 对应18层和34层的基础残差结构
class BasicBlock(nn.Module):
    expansion=1
    def __init__(self,in_channels,out_channels,stride=1,downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1=nn.Conv2d(in_channels=in_channels,out_channels=out_channels
                             ,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn1=nn.BatchNorm2d(out_channels)
        self.relu=nn.ReLU()
        self.conv2=nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,
                             padding=1,bias=False)
        self.bn2=nn.BatchNorm2d(out_channels)
        self.downsample=downsample
    def forward(self,x):
        identity=x
        if self.downsample is not None: #如果下采样函数存在，则对残差分支进行下采样
            identity=self.downsample(identity)
        conv1_out=self.relu(self.bn1(self.conv1(x))) #卷积、bn、激活
        conv2_out=self.bn2(self.conv2(conv1_out)) #卷积、bn，激活函数要加上残差边后再使用
        out_add=conv2_out+identity #卷积输出加上残差边

        out=self.relu(out_add) #激活

        return out

class BasicBlock2(nn.Module):
    expansion=4 #在50层以上的结构中，需要利用1*1卷积进行升维，升的倍数是4倍
    def __init__(self,in_channels,out_channels,stride=1,downsample=None):
        super(BasicBlock2, self).__init__()
        self.conv1=nn.Conv2d(in_channels,out_channels,kernel_size=1,
                             stride=1,bias=False)
        self.bn1=nn.BatchNorm2d(out_channels)
        self.relu=nn.ReLU()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=stride,padding=1,bias=False)
        self.bn2=nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1,
                               stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)
        self.downsample=downsample #残差下采样

    def forward(self,x):
        identity=x #保留残差分支
        if self.downsample is not None:
            identity=self.downsample(identity)

        # 连续的卷积操作
        conv1_out=self.relu(self.bn1(self.conv1(x)))
        conv2_out=self.relu(self.bn2(self.conv2(conv1_out)))
        conv3_out=self.bn3(self.conv3(conv2_out))

        out_add=conv3_out+identity #残差相加

        out=self.relu(out_add)

        return out


class ResNet(nn.Module):
    def __init__(self,block,block_num,num_classes=1000,include_top=True):
        # include_top是为了搭建其他网络时使用，include_top=False时不会采用全连接层，只要卷积主干特征提取网络
        super(ResNet, self).__init__()
        self.include_top=include_top
        self.in_channels=64 #resnet会先经过一个初始卷积，卷积后特征层通道数都是64

        # 初始一个下采样卷积
        self.conv1=nn.Conv2d(3,out_channels=self.in_channels,kernel_size=7,stride=2,
                             padding=3,bias=False)
        # (x-k+2p+1)/s+1，padding=3，使得图片的输出尺寸刚好为原来一般
        self.bn1=nn.BatchNorm2d(self.in_channels)
        self.relu=nn.ReLU()
        # padding=1,使得输出尺寸为原来一半，在默认dilation=1时，maxpool计算公式(h+2*p-k)/s+1
        self.max_pool=nn.MaxPool2d(3,stride=2,padding=1)

        self.layer1=self._make_layer(block,64,block_num[0])
        # stride=2是因为，从第二次开始，第一次卷积会将图片进行下采样
        self.layer2 = self._make_layer(block, 128, block_num[1],stride=2)
        self.layer3 = self._make_layer(block, 256, block_num[2],stride=2)
        self.layer4 = self._make_layer(block, 512, block_num[3],stride=2)

        if self.include_top:
            # 通过平均池化下采样，无论特征层的宽高如何，输出都是1*1宽高
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            # 线性网络，得到分类
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 初始化操作
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self,x):
        x=self.relu(self.bn1(self.conv1(x))) #经过一个初始卷积
        x=self.max_pool(x)
        x=self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x=self.avgpool(x)
            x=torch.flatten(x,1)
            x=self.fc(x)

        return x

    # 生成一个层，一层有多个基本残差block，数量由block_num控制
    def _make_layer(self,block,channels,block_num,stride=1):
        # channels:残差结构中第一层的卷积核的通道数
        downsample=None
        if stride !=1 or self.in_channels != channels * block.expansion:
            # 下采样，通道数变换并且通过stride调整尺寸
            downsample=nn.Sequential(
                nn.Conv2d(self.in_channels,channels*block.expansion,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(channels*block.expansion)
            )
        layers=[]
        # 第一层可能会下采样
        layers.append(block(self.in_channels,channels,downsample=downsample,stride=stride))
        # 若是对于52层以上的，经过当前layers第一个卷积层后，通道数会扩张为当前channels的4倍
        self.in_channels=channels*block.expansion

        for _ in range(1,block_num):
            layers.append(block(self.in_channels,channels))

        return nn.Sequential(*layers)

# 构建18层的resnet
def resnet18(num_classes=1000,include_top=True):
    # 预训练权重链接 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    return ResNet(BasicBlock,[2,2,2,2],num_classes=num_classes,include_top=include_top)

def resnet34(num_classes=1000,include_top=True):
    # 预训练权重链接 https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return ResNet(BasicBlock,[3,4,6,3],num_classes=num_classes,include_top=include_top)

# 可用于构建50层、101层和152层的resnet
# 对于resnet50 输入列表为:[3,4,6,3]
# 对于resnet101 输入列表为:[3,4,23,3]
# 对于resnet152 输入列表为:[3,8,36,3]
def resnet101(num_classes=1000,include_top=True):
    # 预训练权重链接 https://download.pytorch.org/models/resnet50-19c8e357.pth
    return ResNet(BasicBlock2,[3,4,23,3],num_classes=num_classes,include_top=include_top)

if __name__=="__main__":
    net=resnet34()
    test=torch.rand((1,3,214,214))
    out=net(test)
    print(out.shape)