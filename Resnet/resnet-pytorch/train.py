from torchvision import transforms,datasets
import torch
import os
import json
from torch.utils.data import DataLoader
from model import resnet34
from tqdm import tqdm
import sys

def main():
    # 指定设备，device是str类型（字符串）
    device=torch.device("cudn:0" if torch.cuda.is_available() else "cpu")
    print("using {} device".format(device)) #将设备信息输出

    # 数据变换
    # 对于train采用了：随机裁剪再缩放到指定大小、随机水平翻转、转变为Tnesor格式、标准化
    #
    data_transfrom={
        "train":transforms.Compose([transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val":transforms.Compose([transforms.Resize(256),
                                  transforms.CenterCrop(224),
                                  transforms.ToTensor(),
                                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    }
    data_root="D:\datahub\\flower_data_sub" #图片根目录
    # 判断路径是否存在
    assert os.path.exists(data_root), "{} path does not exist.".format(data_root)
    # 训练集
    train_data=datasets.ImageFolder(root=os.path.join(data_root,"train"),
                                    transform=data_transfrom["train"])
    train_num=len(train_data)

    # 由ImageFolder.class_to_idx得到类别索引
    #  {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_class_idx=train_data.class_to_idx # {0: 'daisy', 1: 'dandelion', 2: 'roses', 3: 'sunflowers', 4: 'tulips'}
    # 转变一下字典键与值
    class_dict = dict((v,k) for k,v in flower_class_idx.items()) #{0: 'daisy', 1: 'dandelion', 2: 'roses', 3: 'sunflowers', 4: 'tulips'}
    # write dict into json file，将类别和索引写入josn文件
    josn_str=json.dumps(class_dict,indent=4) # indent=4控制格式缩进，一般为4或2
    with open("class_index.josn","w") as josn_file:
        josn_file.write(josn_str)

    batch_size=16
    # 选择os.cpu_count()、batch_size、8中最小值作为num_workers
    # os.cpu_count()：Python中的方法用于获取系统中的CPU数量。如果系统中的CPU数量不确定，则此方法返回None。
    nw=min([os.cpu_count(),batch_size if batch_size>1 else 0,8])
    print("Using {} dataloader workers every process".format(nw)) #打印使用几个进程

    # 设置训练集dataloader
    train_dataloader=DataLoader(train_data,batch_size=batch_size,
                                shuffle=True,num_workers=nw)
    train_num = len(train_data)
    val_data=datasets.ImageFolder(root=os.path.join(data_root,"val"),
                                  transform=data_transfrom["val"])
    val_num=len(val_data) #验证集长度

    # 设置验证集dataloader
    val_dataloader=DataLoader(val_data,batch_size=batch_size,
                              shuffle=False,num_workers=nw)

    print("Using {} for train,using {} for val".format(train_num,val_num))

    net=resnet34()

    weigth_path="D:\weight_hub\\resnet34-333f7ec4.pth"
    assert os.path.exists(weigth_path),"weight file {} is not exists".format(weigth_path)
    net.load_state_dict(torch.load(weigth_path,map_location=device))

    # change fc layer structure，改变全连接层结构
    # 因为在resnet网络，默认进行1000数的分类，我们花数据集只需要分5类
    inchannels=net.fc.in_features
    net.fc=torch.nn.Linear(inchannels,5)
    net.to(device) #将模型放入设备（cpu或者GPU）中
    print(net.fc)
    # 构造优化器，将参数放入优化器中，设置学习率
    params=[p for p in net.parameters() if p.requires_grad]
    optimizer=torch.optim.Adam(params,lr=0.0001)
    # 损失函数，使用交叉熵损失
    loss_function=torch.nn.CrossEntropyLoss()

    epochs=3
    best_acc=0.0
    save_path="resnet34.pth"
    train_step=len(train_dataloader)
    for epoch in range(epochs):
        # 训练
        net.train()
        running_loss=0.0
        # file = sys.stdout的意思是，print函数会将内容打印输出到标准输出流(即 sys.stdout)
        # train_bar是tqdm用于显示进度
        train_bar=tqdm(train_dataloader,file=sys.stdout)
        data=train_data[0]
        for step,data in enumerate(train_bar):
            images,labels=data
            # images是一个batch的图片，[batcn_size,224,224]
            # labels是每个图片的标签，[batch_szie,]，如[1,0,4],数字代表类别
            optimizer.zero_grad() #先进行梯度清零
            pre=net(images.to(device)) #对类别进行预测
            # print(pre)
            loss=loss_function(pre,labels.to(device))
            loss.backward()
            optimizer.step()
            # loss 统计
            running_loss+= loss.item()

            train_bar.desc="train epoch[{}/{}] loss:{:.3f}".format(epoch+1,epochs,loss)

        #验证模式
        net.eval()
        acc=0.0 #预测正确个数
        with torch.no_grad():
            val_bar = tqdm(val_dataloader, file=sys.stdout)
            for val_d in val_bar:
                val_image,val_label=val_d
                output=net(val_image.to(device))
                # torch.max比较后，第0个是每个最大值，第1个是最大值的下标，所以取第1个
                predict_y=torch.max(output,dim=1)[1]
                acc+=torch.eq(predict_y,val_label.to(device)).sum().item()

                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                       epochs)

        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_step, val_accurate))
        if val_accurate>best_acc:
            best_acc=val_accurate
            torch.save(net.state_dict(),save_path)
    print("finished tarining")



if __name__=="__main__":
    main()