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

## 8.7 V2
v2: 采用8.4的ppt算法，即聚合后再加noise的方案
  
- 修改的点：
  1. cos求和后扰动:几乎相当，略低一点--check
  2. 添加三个扰动
  3. 下发的修改--check..采用全局global grad会影响acc
  4. 上传的修改--跑错了。。需要debug
- 一个奇怪的问题，为什么server_model和global_model没有统一
- 对g_perp_norm的clip非常敏感

- testing: why LOSS explode even when last_grad=0: global_model的参数没找对:没有deepcopy！解决

## 9.18
- test bak版本：聚合过程中添加noise；add noise mean方法中的std没写对：已修改--效果低于sgd
- test新版本：把vector拉平再加noise
  1. 方案1: 在decompose的时候采用平均的cos和norm计算gi perp；使得恢复的值更接近原始
  2. 方案2: 在decompose的时候采用原始的cosi和normi计算gi perp，使得gi perp更加准确--似乎这个方案更好-- testing

## 9.19
- test 回归分层
  1. 检查到底是哪个因素影响了acc
    1）decompose的gi perp normalize后acc降低，why？--不需要分层norm了，分层归一化
    2）norm能否用last grad的norm代替？--可以！
    3）分解后恢复的公式写对了吗: |g| * cos * g_parallel + |g| * sin * g_perp--是对的
- 分层的方案到底应该什么时候分层norm？
  1. 确实应该在层内计算cos，所以垂直分量也在层内取范数
  2. 为什么可以不要norm g_perp同时last_grad norm = [1,1,...]的acc比较高？能不能g_parallel也这么做？
  3. 不要norm g perp 准确的cos和g norm，acc达到85（eps=0.05,perp norm=1）
  4. 不要norm g perp 准确的cos, grad norm=[1]*8 acc=79（eps=0.05,perp norm=1）
  5. 不要norm g perp noisy cos, grad norm=[1]*8 acc=76.85（eps=0.1,perp norm=1）
  
