import sys
import glob
import matplotlib.pyplot as plt
import json

def load_data(x):
    with open(x, "r") as f:
        d = json.loads(f.read())
    return d

def plot_data(data):
    for d in data:
        accs = d["accs"]
        accs_max = [0]
        for a in accs:
            accs_max.append(max(max(accs_max), a))
        accs = accs_max
        w_bits, a_bits = d["W_bits"], d["A_bits"]
        label = "W=%d,A=%d" % (w_bits, a_bits)
        plt.plot(list(range(len(accs))), accs, label=label, linewidth=8)

    plt.xlabel("Epoch", fontsize=22)
    plt.ylabel("Accuracy", fontsize=22)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    #plt.title("Mnist Benefits of Higher Activations (W_bits = %d)" % weights_filter)
    plt.legend(loc="best", fontsize=22)
    plt.tight_layout()
    plt.savefig("mnist_retrain.pdf")

files = glob.glob(sys.argv[1] + "/*")
d = [load_data(x) for x in files]

plot_data(d)
