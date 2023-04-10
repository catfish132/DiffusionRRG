## 原版self.get_extended_attention_mask(batched_inputs)的返回
![img.png](img.png)
## 我的batch input
![img_1.png](img_1.png)
## 原始的batch input
![img_2.png](img_2.png)
## hooks
对于trainer来说hooks注册在其中
每一个hooks有5个对应时刻的函数
trainer在循环中会调用这五个函数
比如trainer调用before_step()时，所有hooks的before_step()函数都会依次调用