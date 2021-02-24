import os
import zipfile
import random
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils.data as Data
from PIL import Image


#########################
# 1.数据准备
#########################

'''
参数配置
'''

train_parameters =  {
    "input_size":[1, 20, 20], #输入图片的shape
    "class_dim":-1, #类别数
    "src_path": r"D:\UserData\lm\Desktop\Recognition number\data\data23617\characterData.zip", #原始数据集路径
    "target_path": r"D:\UserData\lm\Desktop\Recognition number\data\dataset", #解压路径
    "train_list_path": "./train_data.txt", #train_data.txt 路径
    "eval_list_path": "./val_data.txt", #eval_data.txt路径
    "label_dict":{}, #标签字典
    "readme_path":r"D:\UserData\lm\Desktop\Recognition number\data\readme.json",
    "num_epochs":10, #训练轮数
    "train_batch_size": 32, #批次大小
    "learing_strategy":{
        "lr":0.001 #超参数学习率
    }
}

def unzip_data(src_path, target_path):
    '''
    解压数据集，将src_path路径下的zip包解压到target_path路径下
    '''
    if(not os.path.isdir(target_path)):
        z = zipfile.ZipFile(src_path,'r')
        z.extractall(path=target_path)
        z.close()
    else:
        print("文件已解压")
'''
test
'''
# unzip_data(train_parameters["src_path"], train_parameters["target_path"])

def get_data_list(target_path, train_list_path, eval_list_path):
    '''
    生成数据列表
    '''
    #存放所有类别的信息
    class_detail = []
    #获取所有类别保存文件夹的名称
    data_list_path = target_path
    class_dirs = os.listdir(data_list_path)
    if '__MACOSX' in class_dirs:
        class_dirs.remove('__MACOSX')
    #总的图片数量
    all_class_images = 0
    #存放类别标签
    class_label = 0
    #存放类别数目
    class_dim = 0
    #存储要写进eval.txt和train.txt中的内容
    trainer_list=[]
    eval_list=[]
    #读取每个类别
    for class_dir in class_dirs:
        class_dim += 1
        #每个类别的信息
        class_detail_list = {}
        eval_sum = 0
        trainer_sum = 0
        #统计每个类别有多少张图片
        class_sum = 0
        # 获取类别路径
        path = os.path.join(data_list_path,class_dir)
        # print(path)
        img_paths = os.listdir(path)
        for img_path in img_paths:
            if img_path == '.DS_Store':
                continue
            name_path = os.path.join(path,img_path) #每张图片路径
            # print(name_path)
            if class_sum % 10 == 0:  #每10张图片取一个做验证数据
                eval_sum +=1
                eval_list.append(name_path + "\t%d" % class_label + "\n") #测试集
            else:
                trainer_sum += 1
                trainer_list.append(name_path + "\t%d" % class_label + "\n") #训练集
            class_sum +=1
            all_class_images += 1
        #说明json文件中的class_detail数据
        class_detail_list["class_name"] = class_dir #类别名称
        class_detail_list["class_label"] = class_label #类别标签
        class_detail_list["class_eval_images"] = eval_sum #该类数据的测试集数目
        class_detail_list["class_trainer_images"] = trainer_sum #该类数据的训练集数目
        class_detail.append(class_detail_list)
        #初始化标签列表
        train_parameters['label_dict'][str(class_label)] = class_dir
        class_label +=1

    #初始化类别数
    train_parameters["class_dim"] = class_dim
    print(train_parameters)

    #乱序
    # print(eval_list)
    random.shuffle(eval_list)
    with open(eval_list_path, 'a',encoding='utf-8') as f:
        for eval_img in eval_list:
            f.write(eval_img)
    # 乱序
    random.shuffle(trainer_list)
    with open(train_list_path, 'a',encoding='utf-8') as f:
        for train_img in trainer_list:
            f.write(train_img)

    #说明的json文件信息
    readjson = {}
    readjson['all_class_name'] = data_list_path
    readjson['all_class_images'] = all_class_images
    readjson['class_detail'] =class_detail
    jsons = json.dumps(readjson, sort_keys=True, indent=4, separators=(',',':'))
    with open(train_parameters["readme_path"],'w') as f:
        f.write(jsons)
    print('生成数据列表完成！')

'''
test
'''
# get_data_list(train_parameters["target_path"],train_parameters["train_list_path"],train_parameters["eval_list_path"])

def data_reader(file_list):
    '''
    自定义data_reader
    '''
    def reader():
        with open(file_list, 'r') as f:
            lines = [line.strip() for line in f]
            for line in lines:
                imgpath, lab = line.strip().split('\t')
                img = cv2.imread(imgpath)
                img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
                img = np.array(img).astype('float32')
                img = img/255.0
                yield img, int(lab)
    return reader

def data_reader(file_list):
    '''
    自定义data_reader
    '''
    images_list = []  # 文件名列表
    labels_list = []  # 标签列表
    with open(file_list, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f]
        for line in lines:
            imgpath, lab = line.strip().split('\t')
            img = cv2.imread(imgpath)
            if img is None: #判断对象是否是空对象
                continue
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = np.array(img).astype('float32')
            img = img / 255.0
            images_list.append(img)
            labels_list.append(int(lab))
        images_list = torch.Tensor(images_list)
        labels_list = torch.Tensor(labels_list)
        return images_list, labels_list


'''
参数初始化
'''
src_path = train_parameters['src_path']
target_path = train_parameters['target_path']
train_list_path = train_parameters['train_list_path']
eval_list_path = train_parameters['eval_list_path']
batch_size = train_parameters['train_batch_size']

'''
解压原始数据到指定路径
'''
unzip_data(src_path,target_path)

#清空train.txt和eval.txt
with open(train_list_path, 'w') as f:
    f.seek(0)
    f.truncate()
with open(eval_list_path, 'w') as f:
    f.seek(0)
    f.truncate()

#生成数据列表
get_data_list(target_path, train_list_path,eval_list_path)

'''
构造数据提供器
'''
train_data , train_lab = data_reader(train_list_path)
eval_data , eval_lab = data_reader(eval_list_path)


train_dataset = Data.TensorDataset(train_data,train_lab)
eval_dataset = Data.TensorDataset(eval_data,eval_lab)





train_loader = Data.DataLoader(
    dataset=train_dataset,      # torch TensorDataset format
    batch_size=batch_size,      # mini batch size
    shuffle=True,               # random shuffle for training
)

eval_loader = Data.DataLoader(
    dataset=eval_dataset,      # torch TensorDataset format
    batch_size=batch_size,      # mini batch size
    shuffle=True,               # random shuffle for training
)


Batch = 0
Batchs =[]
all_train_accs=[]
def draw_train_acc(Batchs, train_accs):
    title="training accs"
    plt.title(title, fontsize=24)
    plt.xlabel("batch", fontsize=14)
    plt.ylabel("acc", fontsize=14)
    plt.plot(Batchs,train_accs, color='green', label='training accs')
    plt.legend()
    plt.grid()
    plt.show()


all_train_loss =[]
def draw_train_loss(Batchs, train_loss):
    title ="training loss"
    plt.title(title, fontsize=24)
    plt.xlabel("batch", fontsize=14)
    plt.ylabel("loss", fontsize=14)
    plt.plot(Batchs,train_loss, color='red', label='training loss')
    plt.legend()
    plt.grid()
    plt.show()



#########################
# 2.定义模型
#########################
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(     # input shape (batch_size, 1, 20, 20)
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1,              # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),                          # output shape (batch_size, 16,20,20)
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),    # output (batch_size, 16,10, 10)
        )

        self.conv2 = nn.Sequential(     # input shape (batch_size, 16, 10, 10)
            nn.Conv2d(16,32,3,1,1),    # output shape (batch_size, 32,10,10)
            nn.ReLU(),
            nn.MaxPool2d(2),  # output (batch_size, 32,5, 5)
        )
        self.out = nn.Linear(32*5*5, 100)

    def forward(self,input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = x.view(x.size(0), -1) # flatten the output of conv2 to (batch_size, 32*5*5)
        output = self.out(x)
        return output


#########################
# 3.训练模型
#########################
cnn = CNN()
print(cnn)  # net architecture
cnn.cuda()

optimizer = torch.optim.Adam(cnn.parameters(),train_parameters['learing_strategy']['lr'])
loss_func = nn.CrossEntropyLoss()
epochs_num = train_parameters['num_epochs']  # 迭代次数

for epoch in range(epochs_num):
    for step, (x, y) in enumerate(train_loader):

        b_x = torch.unsqueeze(x, dim=1).cuda()
        b_y = y.cuda()
        output = cnn(b_x)           # cnn output
        loss = loss_func(output, b_y.long())   # cross entropy loss
        optimizer.zero_grad()           # clear gradients for this training step
        loss.backward()                 # backpropagation, compute gradients
        optimizer.step()                # apply gradients

        if step != 0 and step % 50 == 0:
            Batch  += 50
            Batchs.append(Batch)
            pred_y = torch.max(output, 1)[1].cuda().data.cpu().numpy()
            accuracy = float((pred_y == b_y.data.cpu().numpy()).astype(int).sum()) / float(b_y.size(0))
            all_train_loss.append(loss.data.cpu().numpy())
            all_train_accs.append(accuracy)

            print("epoch:{},batch_id:{},train_loss:{},train_acc:{}".format(epoch,step,loss.data.cpu().numpy(),accuracy))
# 2 ways to save the net
# torch.save(cnn, 'net.pkl')  # save entire net
torch.save(cnn.state_dict(), 'net_params.pkl')   # save only the parameters
draw_train_acc(Batchs, all_train_accs)
draw_train_loss(Batchs, all_train_loss)

#########################
# 4.模型评估
#########################

accs = []
cnn_test = CNN()
cnn_test.load_state_dict(torch.load('net_params.pkl'))

for step, data in enumerate(eval_loader):
    b_x = torch.unsqueeze(data[0], dim=1)
    b_y = data[1]
    output = cnn_test(b_x)  # cnn output
    pred_y = torch.max(output, 1)[1].data.numpy()
    accuracy = float((pred_y == b_y.data.numpy()).astype(int).sum()) / float(b_y.size(0))
    print(accuracy)



#########################
# 5.使用模型
#########################

# 对车牌图片进行处理，分割出车牌中的每一个字符并保存
license_plate = cv2.imread(r"D:\UserData\lm\Desktop\Recognition number\work\chepai.png")
gray_plate = cv2.cvtColor(license_plate, cv2.COLOR_RGB2GRAY)
ret, binary_plate = cv2.threshold(gray_plate, 175, 255, cv2.THRESH_BINARY)  # ret：阈值，binary_plate：根据阈值处理后的图像数据
# 按列统计像素分布
result = []
for col in range(binary_plate.shape[1]):
    result.append(0)
    for row in range(binary_plate.shape[0]):
        result[col] = result[col] + binary_plate[row][col] / 255
# print(result)
# 记录车牌中字符的位置
character_dict = {}
num = 0
i = 0
while i < len(result):
    if result[i] == 0:
        i += 1
    else:
        index = i + 1
        while result[index] != 0:
            index += 1
        character_dict[num] = [i, index - 1]
        num += 1
        i = index
# print(character_dict)
# 将每个字符填充，并存储
characters = []
for i in range(8):
    if i == 2:
        continue
    padding = (170 - (character_dict[i][1] - character_dict[i][0])) / 2
    # 将单个字符图像填充为170*170
    ndarray = np.pad(binary_plate[:, character_dict[i][0]:character_dict[i][1]], ((0, 0), (int(padding), int(padding))),
                     'constant', constant_values=(0, 0))
    ndarray = cv2.resize(ndarray, (20, 20))
    cv2.imwrite('work/' + str(i) + '.png', ndarray)
    characters.append(ndarray)


def load_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = img.astype('float32')
    img = img[np.newaxis,] / 255.0
    return img


#将标签进行转换
print('Label:',train_parameters['label_dict'])
match = {'A':'A','B':'B','C':'C','D':'D','E':'E','F':'F','G':'G','H':'H','I':'I','J':'J','K':'K','L':'L','M':'M','N':'N',
        'O':'O','P':'P','Q':'Q','R':'R','S':'S','T':'T','U':'U','V':'V','W':'W','X':'X','Y':'Y','Z':'Z',
        'yun':'云','cuan':'川','hei':'黑','zhe':'浙','ning':'宁','jin':'津','gan':'赣','hu':'沪','liao':'辽','jl':'吉','qing':'青','zang':'藏',
        'e1':'鄂','meng':'蒙','gan1':'甘','qiong':'琼','shan':'陕','min':'闽','su':'苏','xin':'新','wan':'皖','jing':'京','xiang':'湘','gui':'贵',
        'yu1':'渝','yu':'豫','ji':'冀','yue':'粤','gui1':'桂','sx':'晋','lu':'鲁',
        '0':'0','1':'1','2':'2','3':'3','4':'4','5':'5','6':'6','7':'7','8':'8','9':'9'}
L = 0
LABEL ={}
for V in train_parameters['label_dict'].values():
    LABEL[str(L)] = match[V]
    L += 1
print(LABEL)


#构建预测动态图过程

model=CNN()#模型实例化
model.load_state_dict(torch.load('net_params.pkl'))
lab=[]
for i in range(8):
    if i==2:
        continue
    infer_imgs = []
    infer_imgs.append(load_image('work/' + str(i) + '.png'))
    infer_imgs = np.array(infer_imgs)
    infer_imgs =torch.Tensor(infer_imgs)
    result=model(infer_imgs)
    lab.append(np.argmax(result.detach().numpy()))
print(lab)
img = Image.open(r"work\chepai.png")
img.show()
for i in range(len(lab)):
    print(LABEL[str(lab[i])],end='')