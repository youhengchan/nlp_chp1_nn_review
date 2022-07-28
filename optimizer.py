# coding: utf-8
import numpy as np
# 【文件状态：已完成，测试通过】
# 2022.6.24 补 2022.6.23 内容，实现 Transformer 的 Optimizer 

# 【文件说明】
# 实现 论文中的 Adam Optimizer 和 dynamic learning rate(动态学习率)，对应论文中 5.3 节 Optimizer
# 除了实现了论文中的 Adam，还实现了 SGD、Momentum、AdaGrad、RMSProp，这都是 Adam 之前的经典 Optimizer
# 也是 Adam 灵感的来源，理解这些 Optimizer 的原理对于理解 Adam 思想有很大好处 

# 【参数说明】
# 1. Transformer 使用了 Adam 优化器，使用的 Adam 优化器 β1 = 0.9, β2 = 0.98, epsilon = 10**-9 
# 2. Transformer 使用了 动态学习率 dynamic_learning_rate = Dmodel**-0.5 * 
# min(step_num**-0.5, step_num * warmup_steps**-1.5),  warmup_steps = 4000

# 【对模型修改说明】
# 由于计算资源的限制，普通电脑可能无法运行 
# Transformer-base： d_model = 512, 
# Transformer_large: d_model = 1024
# 降低 d_model = 256, 实现自定义 transformer-mini: d_model = 256

# 【设计思路：汇报用】
# 深度学习主要是4个步骤，这里实现 4/4 第四个 optimizer 优化器

# 1. 【网络结构定义 layers.py】
# 定义网络结构，将网络中的每一个函数(Function)实现为层(Layer)，每个层有自己的 参数列表(params list)和 
# 梯度列表(grads list)，实现 forward() 和 backward() 方法， 对应正向传播(forward)和反向传播(backward).

# 2. 【正向传播，根据损失函数计算损失LOSS layers.py】
# 每一轮训练都是先正向传播(Forward propagation)，得到损失(Loss)， 然后再反向传播(Back Propagation)
# 对于计算出的损失(Loss) 在正向传播中，输入数据按顺序流过每一个层(Layer), 得到模型输出(Predict value)，
# 并和训练集(Training dataset) 上的教师标签 (Techer Label，一般简记为t) 比对
# 通过损失函数(Loss Function)，传入predict_value 和 teacher label，得到损失(Loss)，简记为L。
# PS：每20步统计一次平均网络的 Loss，并将 Loss 变化情况图形化进行保存(plot.py)，方便写文档

# 3. 【反向传播，计算网络中每个梯度的梯度(grad) layers.py】 
# 使用 步骤2 中计算得到损失L，逐层反向求偏导数(partial derivative)，得到整个网络中的每一个参数的梯度

# ----------> 4. 【参数更新，梯度下降 optimzer.py】<-------------
# 利用 步骤3 中计算得到的梯度 grads 更新网络中的每一个参数，完成一次网络更新，也即一次学习
# 前面3步计算出了网络的每一个参数的梯度，根据更新策略的不同，也对应不同的优化器 Optimizer
# Adam 属于 SGD 的变种，经典常用的优化器有 optimizer (1.SGD, 2.momentum, 3.AdaGrad, 4.RMSProp 5.Adam) 




# 1.【SGD】 Stochasitc Gradient Descent 随机梯度下降
# 【原理】：随机抽取一批数据 batch，并进行梯度下降 GD
# 步骤 1-3 使用 batch 数据算出梯度 grads，将网络中的每个参数 params 都直接减去对应的 
# learning_rate * grads, 这种方案会引起学习过程中的 Loss 的上下波动，且不一定能收敛到全局最优
# 容易在梯度消失的地方停止学习，或者陷入到局部最优
# 这种原始方案学习速率 velocity 完全100%依赖当前计算得到的梯度 grads，如果当前梯度平缓、消失，
# velocity 就趋近 0, 无法继续学习
# PS：当 learning_rate 为正值时，velocity 方向和 grads 方向一致，而 grads 方向为梯度上升方向
# 所以之后 params 梯度下降需要做减法，也就是 params -= velocity
# 【公式】: 
# velocity(本轮学习速度，方向为梯度上升方向) = learning_rate(超参数学习速率，正值) * grads(本轮正反向传播计算出的每个参数对应的梯度，为上升方向)  
# params -= velocity  # 网络所有的参数按照学习速度更新，要梯度下降，所以做减法
class SGD: 
  def __init__(self, learning_rate=0.01): 
    self.learning_rate = learning_rate
  
  # params 和 grads 这里都为 dict 类型, 下同
  def update(self, params, grads):
    for key in params.keys():
      velocity = self.learning_rate * grads[key]
      params[key] -= velocity

# List version SGD
class SGDL:
  def __init__(self, learning_rate=0.01):
    self.learning_rate = learning_rate

  # params and grads are in List format
  def update(self, params, grads):
    for i, _ in enumerate(params):
      velocity = self.learning_rate * grads[i]
      params[i] -= velocity

# 2.【Momentum】动量
# 动量下降法主要为了解决在梯度较小的地方，SGD 学习率很低甚至停止学习的问题
# 【原理】：对于经典的 SGD 算法的改进，在 SGD 算法中更新方案为 params -= learning_rate * grads
# 其中学习速率 learning_rate 通常在开始设置为较大 0.01，后期为 0.001 但更新完全依赖梯度 grads，可以从
# 上面的公式中看出来，只有 grads 较大的时候，才有网络参数的更新，容易陷入到局部最优点 local optimum
# SGD 算法的实际更新速度 velocity = learning_rate * grads, 每次学习，网络更新都是 params -= velocity
# 注意这里的 velocity 是速度而不是速率，是因为 velocity 是有正负的，也就是更新是有方向的

# Momentum 的提出，主要让网络的更新保持一定的“惯性”，所以取了物理中的动量这个名词命名（速度 * 质量）
# 在物理中，动量反应了物体保持自身运动性质的能力，提出该更新策略（或者说有优化器）的作者将这个概念
# 从物理迁移到深度学习中，即让训练过程保持一个参数更新的动力，即使在梯度平稳消失甚至是局部最优点这些 SGD 
# 更新策略难以进行梯度下降到全局最优 global optimum 的情况下，使用动量 Momentum 让网络保持更新的动力，
# 继续进行梯度下降
# Momentum 算法的实际更新速度 velocity 并不再完全依赖于当前的梯度 grads（这和 SGD 不同）
# 而是结合了当前学习步骤 t时刻 的 梯度 grads 和 t-1 即前一时刻的更新速率 latest_velocity 共同计算得到
# 因为引入了上一个时刻的更新速率 latest_velocity, 所以更新速率会保持一种“惯性”，即使是遇到了局部最优点
# 由于“惯性”的存在，依然能一定程度上够保持上一时刻的更新方向和速率（即速度），当然，与物理不同的是，如果没有阻力
# 物体会在太空中一直前行，但是深度学习 LOSS 并不会一直沿着同一个方向一直下降（降到0就是最优），所以会人为引入一个
# 衰减系数 decay 每完成一次学习，前进一步，之前的学习速度带来的影响都会减弱， 同时要考虑到本次梯度计算得到的情况
# 即要用本次计算出的梯度 grads 来引导之后的方向，这部分和 SGD 一样，都是 learning_rate * grads
# PS: 这里 learning_rate * grads 得到的结果为梯度上升方向，而 last_velocity，需要通过调整
# 其中一个的符号将其方向调整为一致。
# 【公式】: 
# if first_time is True: lastest_velocity = 0 (第一次初始化，上一次学习速率设置为0，此时情况和 SGD 相同)
# velocity(梯度上升方向) = decay(衰减系数) * latest_velocity(保持惯性，跳出局部最优) + learning_rate * grads(梯度上升, 同SGD) 
# params -= velocity (目标为梯度下降，做减法)
# latest_velocity = velocity (更新上一次速度为本轮速度)

# PS：在代码实现中，这个衰减系数 decay 被命名为 momentum 来纪念这个工作，所以有时候看到的公式为：
# if first_time is True: latest_velocity = 0
# velocity = momentum * latest_velocity + learning_rate * grads (初始化 velocity 为梯度上升方向)
# params -= velocity 
# latest_velocity = velocity

# PS: 有的代码实现中，在计算 velocity 的时候，在 learning_rate(正数) * grads(梯度上升方向) 前加上了负号
# 即第一次初始化 velocity 的时候，就将 velocity 的方向初始化为梯度下降的方向，那么此时 params += velocity :
# if first_time is True: latest_velocity = 0
# velocity = momentum * latest_velocity - learning_rate * grads (初始化 velocity 为梯度下降方向)
# params += velocity
# last_velocity = velocity

# PS: 上面拆分出 last_velocity 是为了说明算法的思想，在实际的编程中，并不会单独使用一个变量 latest_velocity
# 来记录上一次的速度，而是直接使用一个 velocity 搞定:
# if first_time is True: velocity = 0
# velocity = momentum * velocity + learning_rate * grads
# params -= velocity
# 或者第一次初始化调整 learning_rate * grads 的符号为负号，那么之后就是 params += velocity:
# if first_time is True: velocity = 0
# velocity = momentum * velocity - learning_rate * grads
# params += velocity 

# Momentum
class Momentum:
  def __init__(self, learning_rate=0.01, momentum=0.9):
    self.learning_rate = learning_rate
    self.momentum = momentum
    self.is_first_update = True
    self.velocity = {}

  def init_velocity(self, params):
    for key, value in params.items():
        self.velocity[key] = np.zeros_like(value) # 深度学习中参数都为 np.array, 可能是一个大的权重矩阵 W，也可能是一个偏置向量 b，甚至是单个参数 x，但是这些数据都可以统一表示成 ndarray 类型，这里初始化每个参数的每个位置上的初始更新速率为 0

  def update(self, params, grads):
    if self.is_first_update: 
        self.init_velocity(params)  # 将每一个参数的初始速度都设置为 0
        self.is_first_update = False # 完成初始化，标识设置为 False
    else:  
      for key in params.keys(): 
        self.velocity[key] = self.momentum * self.velocity[key] + self.learning_rate * grads[key] # 计算带动量的 SGD 的更新速度
        params[key] -= self.velocity[key] # 根据计算出的速度更新每一个参数 
  

# 3.【AdaGrad: Adaptive Gradient 自适应梯度】
# 在神经网络的学习过程中，开始梯度较大，可以使用较大的学习率，后面Loss逐步逼近0，梯度变缓
# 为了能够收敛到尽可能接近最优（Loss尽可能接近0），这个时候就要使用较小的学习率，防止Loss波动
# 注意：AdaGrad 是对于每一个参数自适应，也就是 AdaGrad 给每一个参数配备自己的学习速率
# 【原理】：对 SGD 稍微修改，就可以得到 AdaGrad，对比 SGD 和 AdaGrad 更新参数的方法：
# SGD：    velocity = learning_rate * grads;                  params -= velocity
# AdaGrad: velocity = learning_rate * grad * (1/sqrt(decay)); params -= velocity; 
# 其中 decay = decay + Hardmard_prodcut(grads, grads)
# 可以看到，AdaGrad 对于某个参数，会维护该参数历史上所有的梯度值的平方和，然后使用 1/sqrt(decay) 来作为衰减
# 系数。 注意，AdaGrad 会对于每一个参数都适当调整学习率，与此同时进行学习。

# 【公式】
# 记某次学习经过步骤2 正向传播后，得到损失 L，进行步骤3 反向传播时需要更新的参数为 W (为了通用，W设置为矩阵)
# Hardmard_Product 为矩阵对应位置上的位置相乘（不是传统的矩阵乘法）
# 这么做就是矩阵的每个位置都平方，确保结果时非负值，这样训练轮次和 decay 就可以保持正相关关系

# decay = decay + Hardamard_Product(dL/dW, dL/dW)   # W 的每一个位置的衰减系数都随着训练的轮次增长         
# W = W - learning_rate * (1 / sqrt(decay)) * dL/dW   # 使用衰减系数更新网络参数

class AdaGrad:
  def __init__(self, learning_rate=0.01):
    self.learning_rate = learning_rate
    self.decay = None 

  def init_decay(self, params):
    self.decay = {}
    for key, value in params.items():
      self.decay[key] = np.zeros_like(value)


  def update(self, params, grads):
    if self.decay is None:
      self.init_decay(params)
    for key in params.keys(): # 每次 update 都更新衰减系数 
      self.decay[key] += grads[key] * grads[key]
      # 更新完 decay 就可以作为衰减系数用来更新参数
      params[key] -= self.learning_rate * grads[key] * grads[key] / np.sqrt(self.decay[key] + 1e-7)
      # 加上 1e-7 为了防止 self.decay[key] == 0 分母为0

# 4. 【RMSProp: Root Mean Squared Propagation】
# 【原理】RMSProp 是沿着 Momentum 优化算法思路改进的，主要是为了解决原始的 Momentum 的波动过大问题
# 在2 Momentum 中已经介绍了 Momentum 动量改进版的 SGD 可以保持一个梯度下降的“惯性”，使得模型可以快速的
# 朝着最优化方向更新而不会陷入局部最优和平坦地区，但是实验显示这种“惯性”过大，训练过程中会产生较大的波动，
# 使得收敛较为困难，且效果不错，但存在参数更新时波动过大问题。为了保证模型在加快收敛速度的同时保持参数波动平稳
# Hinton 等人提出了自适应的调整，对网络中的参数梯度使用微分平方加权平均数
# 对比原始的 Momentum 和 改进版本的 RMSProp，可以发现，Momentum 的衰减是加性延续动量，而 RMSProp 是乘性衰减
# Momentum 通过加上 momentum 乘以之前的速度，可以使得能部分够延续之前的下降的速度，即保持下降的惯性
# RMSProp 则是通过一个乘性因子来调整梯度下降的速率，这个乘性因子能够保持一定的惯性，但是又能够动态调整缩放梯度影响

#【公式】
# [RMSProp] 
# vt = β * vt-1 + (1 - β) * grads**2 （通常 β = 0.99）
# params -= learning_rate * grads * 1 / sqrt(vt + epsilon) (乘性衰减), epsilon = 10**-7 
# 分析，假设之前vt-1是有下降速度的，落到了平原grads趋近0，或陷入到了局部最优点
# 假设是平原，v 时刻，此时 grads 接近 0，那么 vt = 0.99vt-1 + 0.01 * 0，基本还是会保持之前的系数
# 那么在里面经过了足够多轮卡住 param 更新失效（因为 grads->0），设为 N 轮，在平原后 vt 就会衰减到 0.99**N
# 此时由于因子是作为分母存在，假设卡了300轮，梯度为 0.01，那么 1 / (0.99**300 + 1e-7) = 20.39
# 即此时梯度会被放大 20 倍，那么虽然此时梯度很小，但是又可以更新下去了，所以梯度被放大了，vt 又起来了
# 直到又出现平原又会积累若干轮后持续放大梯度，继续下降。
# 另外，从设计的角度上看，这种待惯性参数本身对于局部最优点有穿透能力，因为在局部降到一个LOSS不变但是LOSS还不为0
# 的时候，此时仍然可以继续下降，此时只是梯度变为趋向0，但只要卡的轮次足够多，就和上面分析平原的原理一样
# vt 总会在足够的轮次之后被放大到几十上百，可以捕捉非常小的梯度特征，继续下降
# PS: 和 Momentum 不同，代码中 momentum 中将 decay 衰减系数命名为 momentum 
# 这里的衰减系数 β 并没有被命名为 rmsprop，而是直接写为 decay=0.99 

# [Momentum] 
# velocity = momentum * velocity (加性衰减) + learning_rate * grads; 
# prams -= velocity

# PS：RMSProp 论文有深度学习之父 Hinton 的挂名，所以有很高知名度
# PS：Hinton 1986 年在 Nature 上发表了 《Learning representations by back-propagating errors》 
# 即所有深度学习算法的基石：反向传播 BP 算法，被称为深度学习之父
# 2012 年 ImageNet 竞赛，Hinton 团队首次使用 GPU 训练的深度学习网络 AlexNet 上场，一战封神
# 引领了接下来 10 年的深度学习的发展

class RMSProp:
  def __init__(self, learning_rate=0.01, decay=0.99):
    self.learning_rate = learning_rate
    self.decay = decay
    self.v = None
	
  def update(self, params, grads):
    if self.v is None: # 第一次更新
      self.v = {}
      for key, value in params.items():
        self.v[key] = np.zeros_like(value)
    else:
      for key, value in params.keys():
        self.v[key] = self.decay * self.v[key] + (1 - self.decay) * grads[key] * grads[key]
        params[key] -= self.learning_rate * grads[key] * 1.0 / np.sqrt(self.v[key] + 1e-7)
	

# 5. 【Adam: Adaptive Moment Estimation 自适应矩估计】 Transformer 论文使用的是这个优化器 
# 另外 Transformer 训练过程中使用了预热 warm-up 机制, 做了一个分段函数来调整学习速率

# 【Transformer Optimizer 和 Learning Rate 参数说明】
# 1. Transformer 使用了 Adam 优化器，使用的 Adam 优化器 β1 = 0.9, β2 = 0.98, epsilon = 10**-9 
# 2. Transformer 使用了 动态学习率 dynamic_learning_rate = Dmodel**-0.5 * 
# min(step_num**-0.5, step_num * warmup_steps**-1.5),  warmup_steps = 4000

# Adam 融合了 Momentum 及其升级版本 RMSProp (保持梯度下降惯性) 和 AdaGrad(动态学习速度衰减) 的优点，2015 提出
# 论文链接 (http://arxiv.org/abs/1412.6980v8) 该论文引用 11w+，是历史上引用最高的机器学习论文之一

# 【思想】
# Adam 其实是吸收了动量系列 RMSProp 的精华部分（可以说公式中 50%）的部分就是直接照搬 RMSProp 的一阶微分量平方 dW^2
# 然后照猫画虎又补充了一个一阶项 dW，有意思的是，这里又开始将衰减系数称为 momentum 而不是 decay 了
# 但是这个照的猫是同时两只猫，即公式的形式照着 RMSProp 抄，保留了 RMSProp 一阶微分项平方项 dW^2 和 分母设计
# 这样就可以和 RMSProp 一样在局部最优点或者是平坦处通过迭代不断放大微小梯度，继续下降。（原理分析见上面的 RMSProp）
# 但是同时他还抄到了动态学习率衰减的 AdaGrad 的精华，AdaGrad 将每次迭代的梯度平方累计去作为一个学习率衰减因子
# 这样可以为每个参数定制自己的学习速率，在学习（训练）后期梯度平缓的时候，能够将学习率降到很低，避免波动，顺利收敛。
# 关于参考这两个算法的思路，这一点在 Adam  的原论文中，作者也在 Introduction 中开门见山地说了
# “We propose Adam, a method for efficient stochastic optimization that only requires first-order gradients with
# little memory requirement. The method computes individual adaptive learning rates for different parameters from 
# estimates of first and second-order moments of the fradients; the name Adam is derived from adaptive moment 
# estimation" 
# Adam 是一种高校地随机优化方法，只需要一阶梯度，对内存的要求很少。该方法通过对梯度的一阶和二阶矩的估计来计算不同参数的个体
# 自适应学习率； Adam 的名字来源于自适应矩估计 
# "Our methods is designed to combine the advantages of two recently popular methods: AdaGrad(Duchi et al., 2011)
# which works well sparse gradient, and RMSProp(Tieleman & Hinton, 2012), which works well on-line and 
# none-stationary setting; imoprtant connections to these and other stochastic optimization methods "
# 我们的方法设计时考虑到结合最近两种流行的方案：AdaGrad(2011)，在稀疏的梯度上工作很好，RMSProp(2012) 在在线和非平稳设置
# 中工作得很好

# 【参数】
# Adam 会设置三个超参数 Hyper Parameters: 
# 1. 第一个 momentum 参数： β1 Adam 论文中标准设置值为 0.9，Transformer 论文中设置为 0.9
# 2. 第二个 momentum 参数： β2 Adam 论文中标准设置值为 0.999，Transformer 论文中设置为 0.98
# 3. 学习率 α，learning_rate 一般取 0.001

# PS：防止分母为0 的 epsilon 设置为 1e-8

# 【公式】按照 Adam 的原版论文，一阶矩m对应的参数为beta1，二阶矩v对应参数为beta2
# 一阶矩放在分子上，就是纯的动量，二阶矩放在分母上，和 RMSProp 一致
# learning_rate = 0.001
# In iteration times t:
# m = beta1 * m + (1 - beta1) * dW          # beta1 = 0.9
# v = beta2 * v + (1 - beta2) * dw**2       # beta2 = 0.98 Transformer 取值不是 0.999
# m_corrected = m / (1 - beta1**t)          # 修正偏移影响
# v_corrected = v / (1 - beta2**t)          # 修正偏移影响
# W = W - learning_rate * m_corrected / ( sqrt(v_corrected) + epsilon) 其中 epsilon = 10**-8, Transformer 论文中使用了更高精度的 1e-9

# PS: 原论文提出了一种可以提高计算效率 Efficiency，但是会降低清晰度 Clarity 的方案：
# 在论文的 第二节 Algorithm 中指出了，可以通过将循环中最后的三行修改为下面的两行实现：
# learning_rate_t = learning_rate * sqrt(1 - beta2**t) / (1 - beta1**t)
# W -= learning_rate_t * m / (sqrt(v) + 1e-8)
# 即两步单独修正偏移影响得到 m_corrected 和 v_corrected 的步骤通一个学习率修正 learning_rate_t 一步完成

#  一阶微分平方（二阶矩）做为分母，这个是延续了 RMSProp 的设计思路（一模一样） 
#  一阶微分（一阶矩）作为分子，如果去掉分母，就是一个加权和为1的原版 Momentum

class Adam:
  def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.98, epsilon=1e-9, in_fast_mode=True):
    self.epsilon = epsilon
    self.learning_rate = learning_rate
    self.m = None # 梯度一阶矩，分子
    self.m_corrected = None # 修正值
    self.beta1 = beta1 # 滑动参数
    self.v = None # 梯度二阶矩，分母
    self.v_corrected = None # 修正值
    self.beta2 = beta2 # 滑动参数
    self.t = 0 # 迭代次数

  # 默认开启 fast_mode 提高训练速度，损失一定精度
  def update(self, params, grads, in_fast_mode=True):
    if self.m is None:
      self.m, self.v = {}, {}
      if not in_fast_mode:
        self.m_corrected, self.v_corrected = {}, {}
      for key, value in params.items():
        self.m[key] = np.zeros_like(value)
        self.v[key] = np.zeros_like(value)
        if not in_fast_mode:
          self.m_corrected[key] = np.zeros_like(value)
          self.v_corrected[key] = np.zeros_like(value)
          
    self.t += 1 # 更新次数加上1
    if in_fast_mode:
      learning_rate_t = self.learning_rate * np.sqrt(1.0 - self.beta2**self.t) / (1.0 - self.beta1**self.t)
      # 更新所有参数
      for key in params.keys():
        self.m[key] = self.beta1 * self.m[key] + (1.0 - self.beta1) * grads[key]
        self.v[key] = self.beta2 * self.v[key] + (1.0 - self.beta2) * grads[key] * grads[key]
        params[key] -= learning_rate_t * self.m[key] / (np.sqrt(self.v[key]) + self.epsilon)
    else:
      for key in params.keys():
        self.m[key] = self.beta1 * self.m[key] + (1.0 - self.beta1) * grads[key]
        self.v[key] = self.beta2 * self.v[key] + (1.0 - self.beta2) * grads[key] * grads[key]
        self.m_corrected[key] = self.m[key] / (1.0 - self.beta1**self.t)
        self.v_corrected[key] = self.v[key] / (1.0 - self.beta2**self.t)
        params[key] -= self.learning_rate * self.m_corrected[key] / (np.sqrt(self.v_corrected[key]) + self.epsilon)

      
# 主要测试 transformer 使用的 Adam Optimizer 工作情况
def test():
  params = {'W1': np.array([[6, 6, 6, 6, 6], [6, 6, 6, 6, 6]], dtype=float), 'b': np.array([1, 2, 3, 4, 5], dtype=float)}
  grads = {'W1': np.array([[0.5, 0.5, 0.5, 0.5, 0.5], [0.6, 0.6, 0.6, 0.6, 0.6]]), 'b': np.array([0.5, 0.6, 0.7, 0.8, 0.9])}
  adam_optimizer = Adam()
  for learning_steps in range(1000):
    adam_optimizer.update(params, grads)
  print(f"params: {params}\n"
        f"grads: {grads}")


if __name__ == "__main__":
  test()

'''
测试输出
第一组[learning_step=100]
params: {'W1': array([[5.9, 5.9, 5.9, 5.9, 5.9],
       [5.9, 5.9, 5.9, 5.9, 5.9]]), 'b': array([0.9, 1.9, 2.9, 3.9, 4.9])}
grads: {'W1': array([[0.5, 0.5, 0.5, 0.5, 0.5],
       [0.6, 0.6, 0.6, 0.6, 0.6]]), 'b': array([0.5, 0.6, 0.7, 0.8, 0.9])}
第二组[learning_step=10000000]
params: {'W1': array([[-9993.99998207, -9993.99998207, -9993.99998207, -9993.99998207,
        -9993.99998207],
       [-9993.99998382, -9993.99998382, -9993.99998382, -9993.99998382,
        -9993.99998382]]), 'b': array([-9998.99998207, -9997.99998382, -9996.99998429, -9995.9999856 ,
       -9994.99998947])}
grads: {'W1': array([[0.5, 0.5, 0.5, 0.5, 0.5],
       [0.6, 0.6, 0.6, 0.6, 0.6]]), 'b': array([0.5, 0.6, 0.7, 0.8, 0.9])}
'''


# 【总结】
# 最常见的5种 Optimizer 1. SGD  2. Momentum 3. AdaGrad 4. RMSProp 5. Adam
# 目前最常用的就是原始的 SGD 和 Adam（融合了动量方法 Momentum 及其改进版 RMSProp 和 速率衰减 AdaGrad）
# 但是目前对于几十亿 GPT2 1.5B GPT2 8B 几千亿 GPT3 175B 万亿 Swich Transformer 1300B 参数的模型，有更新的分布式优化器 LAMB 等