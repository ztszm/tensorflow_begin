#设置待识别的图片
import cv2
import numpy as np
from keras import layers, Input
from keras import Sequential
from keras.src.layers import Conv2D, MaxPooling2D
import tensorflow as tf

def create_model():
    model = Sequential()
    input_layer = Input(shape=(24, 24, 1))
    model.add(input_layer)
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(15))
    model.add(layers.Dense(15, activation='softmax'))
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model
img1=cv2.imread('1.png',0)
img2=cv2.imread('2.png',0)
imgs=np.array([img1,img2])
#构建模型
model=create_model()
#加载前期训练好的权重
model.load_weights('checkpoint/char_checkpoint.weights.h5')
#读出图片分类
class_name=np.load('class_name.npy')
#预测图片，获取预测值
predicts=model.predict(imgs)
results=[]#保存结果的数组
for predict in predicts:#遍历每一个预测结果
    index=np.argmax(predict)#寻找最大值
    result=class_name[index]#取出字符
    results.append(result)
print(results)