**本文来自公众号CV技术指南，欢迎用于个人学习，严禁用于商业行为。**
 
**欢迎关注公众号CV技术指南，专注于计算机视觉的技术总结、最新技术跟踪、经典论文解读、CV招聘信息。**


>   **前言** 本文给大家分享一份我用的特征图可视化代码。



#### **写在前面的话**

------

**特征图可视化是很多论文所需要做的一份工作，其作用可以是用于证明方法的有效性，也可以是用来增加工作量，给论文凑字数**。

具体来说就是可视化两个图，使用了新方法的和使用之前的，对比有什么区别，然后看图写论文说明新方法体现的作用。

吐槽一句，有时候这个图 论文作者自己都不一定能看不懂，虽然确实可视化的图有些改变，但并不懂这个改变说明了什么，反正就吹牛，强行往自己新方法编的故事上扯，就像小学一年级的作文题--看图写作文。

之前知乎上有一个很热门的话题，如果我在baseline上做了一点小小的改进，却有很大的效果，这能写论文吗？

这种情况最大的问题就在于要如何写七页以上，那一点点的改进可能写完思路，公式推理，画图等内容才花了不到一页，剩下的内容如何搞？可视化特征图！！！

这一点可以在我看过的甚多论文上有所体现，反正我是没看明白论文给的可视化图，作者却能扯那么多道道。这应该就是用来增加论文字数和增加工作量的。

总之一句话，**可视化特征图是很重要的工作，最好要会**。



#### **初始化配置**

------

这部分先完成加载数据，修改网络，定义网络，加载预训练模型。

##### **加载数据并预处理**

这里只加载一张图片，就不用通过classdataset了，因为classdataset是针对大量数据的，生成一个迭代器一批一批地将图片送给网络。但我们仍然要完成classdataset中数据预处理的部分。

数据预处理所必须要有的操作是调整大小，转化为Tensor格式，归一化。至于其它数据增强或预处理的操作，自己按需添加。

```python
def image_proprecess(img_path):
    img = Image.open(img_path)
    data_transforms = transforms.Compose([
        transforms.Resize((384, 384), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    data = data_transforms(img)
    data = torch.unsqueeze(data,0)
    return data
```

这里由于只加载一张图片，因此后面要使用torch.unsqueeze将三维张量变成四维。



##### **修改网络**

假如你要可视化某一层的特征图，则需要将该层的特征图返回出来，因此需要先修改网络中的forward函数。具体修改方式如下所示。

```python
def forward(self, x):
    x = self.model.conv1(x)
    x = self.model.bn1(x)
    x = self.model.relu(x)
    x = self.model.maxpool(x)
    feature = self.model.layer1(x)
    x = self.model.layer2(feature)
    x = self.model.layer3(x)
    x = self.model.layer4(x)
    return feature,x
```



##### **定义网络并加载预训练模型**

```python
def Init_Setting(epoch):
    dirname = '/mnt/share/VideoReID/share/models/Methods5_trial1'
    model = siamese_resnet50(701, stride=1, pool='avg')
    trained_path = os.path.join(dirname, 'net_%03d.pth' % epoch)
    print("load %03d.pth" % epoch)
    model.load_state_dict(torch.load(trained_path))
    model = model.cuda().eval()
    return model
```

这部分需要说明的是最后一行，要将网络设置为推理模式。



#### **可视化特征图**

------

这部分主要是将特征图的某一通道转化为一张图来可视化。

```python
def visualize_feature_map(img_batch,out_path,type,BI):
    feature_map = torch.squeeze(img_batch)
    feature_map = feature_map.detach().cpu().numpy()

    feature_map_sum = feature_map[0, :, :]
    feature_map_sum = np.expand_dims(feature_map_sum, axis=2)
    for i in range(0, 2048):
        feature_map_split = feature_map[i,:, :]
        feature_map_split = np.expand_dims(feature_map_split,axis=2)
        if i > 0:
            feature_map_sum +=feature_map_split
        feature_map_split = BI.transform(feature_map_split)

        plt.imshow(feature_map_split)
        plt.savefig(out_path + str(i) + "_{}.jpg".format(type) )
        plt.xticks()
        plt.yticks()
        plt.axis('off')

    feature_map_sum = BI.transform(feature_map_sum)
    plt.imshow(feature_map_sum)
    plt.savefig(out_path + "sum_{}.jpg".format(type))
    print("save sum_{}.jpg".format(type))
```

这里一行一行来解释。

1. 参数img_batch是从网络中的某一层传回来的特征图，BI是双线性插值的函数，自定义的，下面会讲。

2. 由于只可视化了一张图片，因此img_batch是四维的，且batchsize维为1。第三行将它从GPU上弄到CPU上，并变成numpy格式。

3. 剩下部分主要完成将每个通道变成一张图，以及将所有通道每个元素对应位置相加，并保存。



##### **双线性插值**

------

由于经过多次网络降采样，后面层的特征图往往变得只有7x7,16x16大小。可视化后特别小，因此需要将它上采样，这里采样的方式是双线性插值。因此，这里给一份双线性插值的代码。

```python
class BilinearInterpolation(object):
    def __init__(self, w_rate: float, h_rate: float, *, align='center'):
        if align not in ['center', 'left']:
            logging.exception(f'{align} is not a valid align parameter')
            align = 'center'
        self.align = align
        self.w_rate = w_rate
        self.h_rate = h_rate

    def set_rate(self,w_rate: float, h_rate: float):
        self.w_rate = w_rate    # w 的缩放率
        self.h_rate = h_rate    # h 的缩放率

    # 由变换后的像素坐标得到原图像的坐标    针对高
    def get_src_h(self, dst_i,source_h,goal_h) -> float:
        if self.align == 'left':
            # 左上角对齐
            src_i = float(dst_i * (source_h/goal_h))
        elif self.align == 'center':
            # 将两个图像的几何中心重合。
            src_i = float((dst_i + 0.5) * (source_h/goal_h) - 0.5)
        src_i += 0.001
        src_i = max(0.0, src_i)
        src_i = min(float(source_h - 1), src_i)
        return src_i
    # 由变换后的像素坐标得到原图像的坐标    针对宽
    def get_src_w(self, dst_j,source_w,goal_w) -> float:
        if self.align == 'left':
            # 左上角对齐
            src_j = float(dst_j * (source_w/goal_w))
        elif self.align == 'center':
            # 将两个图像的几何中心重合。
            src_j = float((dst_j + 0.5) * (source_w/goal_w) - 0.5)
        src_j += 0.001
        src_j = max(0.0, src_j)
        src_j = min((source_w - 1), src_j)
        return src_j

    def transform(self, img):
        source_h, source_w, source_c = img.shape  # (235, 234, 3)
        goal_h, goal_w = round(
            source_h * self.h_rate), round(source_w * self.w_rate)
        new_img = np.zeros((goal_h, goal_w, source_c), dtype=np.uint8)

        for i in range(new_img.shape[0]):       # h
            src_i = self.get_src_h(i,source_h,goal_h)
            for j in range(new_img.shape[1]):
                src_j = self.get_src_w(j,source_w,goal_w)
                i2 = ceil(src_i)
                i1 = int(src_i)
                j2 = ceil(src_j)
                j1 = int(src_j)
                x2_x = j2 - src_j
                x_x1 = src_j - j1
                y2_y = i2 - src_i
                y_y1 = src_i - i1
                new_img[i, j] = img[i1, j1]*x2_x*y2_y + img[i1, j2] * \
                    x_x1*y2_y + img[i2, j1]*x2_x*y_y1 + img[i2, j2]*x_x1*y_y1
        return new_img
#使用方法
BI = BilinearInterpolation(8, 8)
feature_map = BI.transform(feature_map)
```



##### **main函数流程**

------

上面介绍了各个部分的代码，下面就是整体流程。比较简单。

```python
imgs_path = "/path/to/imgs/"
save_path = "/save/path/to/output/"
model = Init_Setting(120)
BI = BilinearInterpolation(8, 8)

data = image_proprecess(out_path + "0836.jpg")
data = data.cuda()
output, _ = model(data)
visualize_feature_map(output, save_path, "drone", BI)
```



#### **可视化效果图**

------

![图片](https://mmbiz.qpic.cn/mmbiz_png/V2E1ll6kaTVqAwbeVAGXfmmlBRwZcbMXLoBIwhvU8SkrFicuricQQZy4CwG5DfqF4ff16wNUNuNSNSicIG2l6icDbg/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)


**欢迎关注公众号CV技术指南，专注于计算机视觉的技术总结、最新技术跟踪、经典论文解读、CV招聘信息。**
