"""Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
"""

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

# batch_size 每个批次跑的样本数量，样本太小会导致训练慢，过拟合。太大会导致欠拟合，如果样本数量是60000，那么一个
# 迭代跑的批次就是60000/128
batch_size = 256

# 0-9手写数字一个有10个类别
num_classes = 10

# 12次完整迭代
epochs = 2

# 输入图像的大小
img_rows, img_cols = 28, 28

# 导入数据
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 数据有的是颜色通道数在前，有的是在后
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

# 数据变成浮点更精确
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# 数据正则化
x_train /= 255
x_test /= 255


print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# 把数字0-9的类别变成二进制，方便训练
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# 构建模型
model = Sequential()

# 添加第一个2D卷积层，32个滤波器，滤波器/卷积核大小为3X3
# 激活函数选用relu
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))

# 添加第二个2D积卷层，64个滤波器，滤波器/卷积核大小为3X3
model.add(Conv2D(64, (3, 3), activation='relu'))

# 添加第一个池化层/采样层，用MaxPolling函数，采样大小为2X2
model.add(MaxPooling2D(pool_size=(2, 2)))

# 对池化层/采样层，采用0.25的比例的丢弃率
model.add(Dropout(0.25))

# 展平层，展平所有像素，从[28*28]->[748]
model.add(Flatten())

# 添加第一个全连接层，128个神经元(也就是输出为128个)，激活函数使用relu
model.add(Dense(128, activation='relu'))

# 对全连接层，采用0.5的比例的丢弃率
model.add(Dropout(0.5))

# 添加第二个全连接层，10个神经元(也就是输出为10个)，激活函数使用softmax
model.add(Dense(num_classes, activation='softmax'))

# 编译模型，使用的损失函数为cross entropy,使用的梯度下降算法为Ada delta
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph2', histogram_freq=0, write_graph=True, write_images=True)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# 开始培训/优化模型，设置跑12次迭代， 每个迭代跑128个样本
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[tbCallBack])

# 用测试数据看看预测的精度是多少？
score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])