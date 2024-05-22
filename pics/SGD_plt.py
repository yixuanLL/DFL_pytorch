import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
N=50

start = -1
end = 3
plt.switch_backend('agg')
def get_loss_map(loss_fn, x, y):
    """Maps the loss function on a 100-by-100 grid between (-5, -5) and (8, 8)."""
    losses = [[0.0] * 201 for _ in range(201)]
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)
    for wi in range(201):
        for wb in range(201):
          w = start + (end-start) * wi / 200.0
          b = start + (end-start) * wb / 200.0
          ywb = x * w + b
          losses[wi][wb] = loss_fn(ywb, y).item()
    # return list(reversed(losses))  # Because y will be reversed.
    print(np.min(losses))
    return list(losses)

def SGD_visualization():
    """
    z = 0.75*x+y+2+noise
    """
    np.random.seed(20240517)
    n = N
    x = np.array(np.random.randn(n), dtype=np.float32)
    y = np.array(0.75 * x**1 + 1.0 * x + 2.0 + 0.5 * np.random.randn(N), dtype=np.float32)

    model = torch.nn.Linear(1,1)
    model.weight.data.fill_(0)
    model.bias.data.fill_(0)

    loss_fn = torch.nn.MSELoss()
    learning_rate = 0.12
    epochs = 20
    batch_size = 5
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    models = [[model.weight, model.bias.item()]]
    pre_ = (model.weight.item(),model.bias.item())
    next_ = (0,0)
    fig, ax = plt.subplots()
    axins = ax.inset_axes((0.65, 0.105, 0.3, 0.3))
    for epoch in range(epochs):
        inputs = torch.from_numpy(x[epoch:epoch+batch_size]).requires_grad_().reshape(-1,1)
        labels = torch.from_numpy(y[epoch:epoch+batch_size]).reshape(-1, 1)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        # next_ = (model.weight.grad.data[0][0],model.weight.grad.data[0][1])
        optimizer.step()
        next_= (model.weight.item(),model.bias.item())
        ax.arrow(*pre_, *np.array(next_) - np.array(pre_), head_width=0.08, head_length=0.08, fc='b', ec='b')
        if epoch >=10:
            axins.arrow(*pre_, *np.array(next_) - np.array(pre_), head_width=0.03, head_length=0.03, fc='b', ec='b')
        pre_=next_
        # print(model.weight, model.bias.item())
        models.append([model.weight, model.bias.item()])
        
    losses = get_loss_map(loss_fn,x,y)
    levels = np.linspace(np.min(losses)+1,np.max(losses),8)
    print(levels)
    # ct = ax.contour(np.linspace(-5,8,201),np.linspace(-5,8,201),losses, norm=LogNorm())
    ct = ax.contour(np.linspace(start,end+1,201),np.linspace(start,end,201),losses,levels=levels)
    ax.clabel(ct, inline=1, fontsize=10)
    ax.scatter(1.75,2,marker='*',color='red')
    axins.scatter(1.75,2,marker='*',color='red')
    axins.set_xlim(1.2,2)
    axins.set_ylim(1.7,2.3)
    # ct_in = axins.contour(np.linspace(start,end,201),np.linspace(start,end,201),losses,levels=[0.5,1])
    # axins.clabel(ct_in, inline=1, fontsize=10)
    # mark_inset(ax, axins, loc1=1, loc2=2, fc="none", ec='k', lw=1)
    ax.set_xlabel('Weight', fontsize=16)
    ax.set_ylabel('Bias', fontsize=16)
    ax.set_xlim(start, end+1)
    ax.set_ylim(start, end)
    ax.set_aspect('equal')  # 确保x和y的比例相同
    # 显示图像
    plt.show()
    root_path = '/local/scratch/yliu270/workspace/DFL_pytorch/pics/'
    file_list_path = root_path
    file_name = 'SGD.pdf'
    plt.savefig(file_list_path+file_name, dpi=600)
    plt.close()
SGD_visualization()