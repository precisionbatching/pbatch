import sys
import matplotlib.pyplot as plt

def pbatch(m, n, weight_bits=8, act_bits=8):
    n_bytes = m*n*weight_bits/8 + n*act_bits/8
    n_ops = m*n*act_bits*weight_bits*2
    intensity = n_ops/n_bytes
    return intensity

def standard_inference(m, n, weight_bits):
    n_bytes = m*n*weight_bits/8 + n*weight_bits/8
    n_ops = 2*m*n
    intensity = n_ops/n_bytes
    return intensity

def plot():
    xs = [4, 8, 16, 32]
    ys = [pbatch(1024, 1024, weight_bits=32, act_bits=x) for x in xs]
    plt.plot(xs, ys, label="Pbatch", marker="o", linewidth=4, color="r", markersize=10)

    ys = [standard_inference(1024, 1024, 32) for x in xs]
    plt.plot(xs, ys, label="32-bit inference", linewidth=4, color="blue", marker="x", markersize=10)

    ys = [standard_inference(1024, 1024, 16) for x in xs]
    plt.plot(xs, ys, label="16-bit inference", linewidth=4, color="g", marker="v", markersize=10)

    ys = [standard_inference(1024, 1024, 8) for x in xs]
    plt.plot(xs, ys, label="8-bit inference", linewidth=4, color="yellow", marker="^", markersize=10)

    ys = [standard_inference(1024, 1024, 4) for x in xs]
    plt.plot(xs, ys, label="4-bit inference", linewidth=4, color="black", marker="*", markersize=10)

    plt.xlabel("Pbatch Activation Bits", fontsize=22)
    plt.ylabel("Ops/Bytes", fontsize=22)
    plt.yscale("log", base=2)
    plt.legend(loc="best", fontsize=14)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.savefig("intensity.pdf")

print(standard_inference(1024, 1024, 32))
print(standard_inference(1024, 1024, 16))
print(standard_inference(1024, 1024, 8))
print(standard_inference(1024, 1024, 4))
print(standard_inference(1024, 1024, 1))

print("--")

print(pbatch(1024, 1024, weight_bits=4, act_bits=4))
print(pbatch(1024, 1024, weight_bits=4, act_bits=8))
print(pbatch(1024, 1024, weight_bits=4, act_bits=16))

plot()
