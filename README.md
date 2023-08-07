# DFL_pytorch
directional decomposition SGD in FL, by pytorch

## 8.4
# 验证方向分解的代码是否正确
- 当不加noise、rate=1时，dr的acc是否等于sgd的acc？clients=2 epoch=2 round=10
dr acc 0.7387,0.8245,0.8709,0.906,0.9167
sgd acc 0.9103,0.895,0.9504,0.9601,0.9621
- dr的acc和sgd的acc为什么不一样？？
  1. 是不是分解错了：是nomalize last——grad错了：改完是0.7425,0.8247,0.8762,0.9056,0.9185
  2. 是不是reverse错了:没错
  3. **是norm限制了acc！norm调成10就没问题！！**
  - DPOptimizer本身的acc没那么高：DPoptimizer中如果不加noise：0.7462,0.8216,0.8772,0.9093,0.9214
  - norm调成10: DPoptimizer：0.916,0.9444,0.958,0.9638,0.9659; DrDPoptimizer: 0.908,0.9343,0.9551,0.9624,0.9632

- dr-topk的影响有多大？
  - rate=1时：0.913,0.9447,0.953,0.9619,0.9618 （g_prep的clip会进一步影响）
  - rate=0.001时影响依然很小（lr调到0.01时）

- clip对dr-topk的影响又多大？
  - g_perp的影响非常大 至少要1
  - **但g的norm可以调到100， acc很高(90以上)**

- topk baseline:topk_optimizer
  - 收到g norm的影响很大

- 实现目前ppt上的算法

--------------
- 添加噪声以后呢？

- 不同model可以采用不同的clip/lr

- 新的算法
  
