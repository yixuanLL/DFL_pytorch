
import matplotlib.pyplot as plt

def grad_plot(log):
    global_round = len(log)
    clean = []
    noisy = []
    estimate = []
    rounds = 0
    for i in range(global_round):
        local_round = len(log[i])
        for j in range(local_round):
            c, n, e = log[i][j]
            clean.append(c[0].reshape(-1)[0])
            noisy.append(n[0].reshape(-1)[0])
            estimate.append(e[0].reshape(-1)[0])
            rounds += 1 

    plt.switch_backend('agg')
    r = range(rounds)
    plt.plot(r, clean, 'b', label='clean grad')
    plt.plot(r, noisy, 'g', label='noisy grad')
    plt.plot(r, estimate, 'c', label='estimate grad')

    plt.ylabel('Gradients')
    plt.xlabel('Rounds')
    plt.legend(loc='lower right', fontsize=8)

    # plt.title('FLamby $\epsilon$=0.5', fontsize=9)
    plt.show()
    root_path = '/home/yliu270/workspace/DFL_pytorch/'
    plt.savefig(root_path+'grad.png', dpi=600)
    plt.close()