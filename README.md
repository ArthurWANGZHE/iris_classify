首先通过pandas、numpy、matlablib，导入数据，并绘制散点图，对于任务有了一个大概的认知。（刚拿到的时候并不很明确要做什么）

导入数据

![image-20230116210907906](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20230116210907906.png)

对数据进行分类整理，并绘图

![image-20230116211116626](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20230116211116626.png)

![image-20230116211004733](file://C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20230116211004733.png?lastModify=1674188032)

一开始看数据介绍里，是给了花的两个部分（？）每个部分各两组数据，于是分开画了两张图。

![image-20230116211130126](file://C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20230116211130126.png?lastModify=1674188097)

![image-20230116211012111](file://C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20230116211012111.png?lastModify=1674188104)

（然而好像并没有什么用。。不过确实帮助我理解了要求。。

首先尝试的事KNN算法，选择KNN算法的原因是，计算上比较简单，比较容易理解（感觉比较容易写？）

第一次尝试没有调用sklearn包

导入数据以后将花名，赋值0,1,2.并没有想到什么好的随机方法于是就比较复杂，分成了一个训练集和测试集。一开始想的事训练集里三种花平均分布，于是这一步就写的比较复杂

![image-20230120121827383](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20230120121827383.png)

第一次尝试里，距离计算也比较草率？反正就是跟着浅薄的数学知识走。没有想到要去定义函数什么，就用了两个for循环硬写。

![image-20230120122252641](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20230120122252641.png)

在最后统计结果的时候，也只是在k个近邻里取平均值作为判断结果。（结果不很令人满意。。正确率只有34%

![image-20230120122504708](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20230120122504708.png)

第二次尝试就想先验证一下KNN算法，于是直接调用了sklearn的KNN。正确率在97%（那问题一定是我写的有问题而不是算法的问题

![image-20230120122720592](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20230120122720592.png)

然后就开始了第三次尝试，重新写了一遍KNN

第一个变化是划分训练集测试集的时候并没有刻意让三种花平均分布

![image-20230120122906763](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20230120122906763.png)

第二个变化是，虽然没有改变计算逻辑，但给距离计算写了个函数（算优化代码？

![image-20230120122945793](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20230120122945793.png)

第三个，在统计距离的时候也稍微优化了一下代码（本来是调用了什么库。。。）现在是用sorted函数

![image-20230120123132831](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20230120123132831.png)

第四个，在取最后判断的时候，也从直接平均，改为了距离加权平均。

![image-20230120123236207](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20230120123236207.png)

最终正确率在91-97

![image-20230120123317780](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20230120123317780.png)

可以改进的地方：

在最后一步，从加权平均的结果到最终结果的时候，是将0-2划分为（0,0.5],(0.5,1.5],(1.5,2]这样三个区间对应三种结果。感觉这样划分有点不合理 但又不知道怎么划分？

关于KNN的学习了解结束~

第四次尝试，SVM算法
根据理解进行计算，第一次尝试的结果不很满意
![img.png](img.png)
感觉貌似是有什么地方写错了。。。

用sklearn验证一下，并阅读了相关源码，结果如下：

![image-20230120123625798](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20230120123625798.png)

