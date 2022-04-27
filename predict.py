import torch
from torchvision import transforms
import os
from PIL import Image
import matplotlib.pyplot as plt
from model import resnet34
import json

def main():
    device=torch.device("cudu:0" if torch.cuda.is_available() else "cpu")
    image_path="D:\datahub\\flower_data\\test_daisy.jpg"
    assert os.path.exists(image_path),"image {} does not exist.".format(image_path)

    # 数据预处理操作，要和train.py中的验证集预处理操作相同
    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # 读取图片
    image=Image.open(image_path)
    plt.imshow(image)  #展示图片

    img=data_transform(image) #对图片进行预处理操作
    # 扩展一个batch维度，加在第0维度，则img的shape变成[1,3,224,224]
    img=torch.unsqueeze(img,dim=0)

    # 读取类别文件json
    json_path="class_index.josn"
    assert os.path.exists(json_path), "json {} does not exist.".format(json_path)
    json_file=open(json_path)
    cls_index=json.load(json_file)
    # 因为不用导入预训练权重，所以直接将分类类别赋值，由训练出的resnet34.pth类别决定
    net=resnet34(num_classes=5)
    weight_path="resnet34.pth"
    assert os.path.exists(weight_path), "weigth {} does not exist.".format(weight_path)
    net.load_state_dict(torch.load(weight_path,map_location=device))
    net.eval()
    with torch.no_grad():
        output=torch.squeeze(net(img.to(device))).cpu()
        pre=torch.softmax(output,dim=0) #将神经网络预测的值经过softmax变成概率
        cls=torch.argmax(pre).numpy() #预测的类别的索引

    print_res = "class: {}   prob: {:.3}".format(cls_index[str(cls)],
                                                 pre[cls].numpy())
    plt.title(print_res) #将结果写在图片的title上
    for i in range(len(pre)):
        print("class: {:10} prob: {:.3}".format(cls_index[str(i)],
                                                  pre[i].numpy()))
    plt.show()




if __name__=="__main__":
    main()