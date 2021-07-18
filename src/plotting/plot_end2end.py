import sys
import re
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

# Very slow for many datapoints.  Fastest for many costs, most readable
def is_pareto_efficient_dumb_max(costs):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_efficient = np.ones(costs.shape[0], dtype = bool)
    for i, c in enumerate(costs):
        is_efficient[i] = np.all(np.any(costs[:i]<c, axis=1)) and np.all(np.any(costs[i+1:]<c, axis=1))
    return is_efficient

# Negate = False for ppl
def get_pareto_points(points, negate=False):
    if negate:
        points_to_use = [(-x[0],x[1]) for x in points]
    else:
        points_to_use = points
    pareto_front = is_pareto_efficient_dumb_max(np.array(points_to_use))
    return [points[i] for i in range(len(points)) if pareto_front[i]]

def read_data(fpath):

    def extract_weight_precision(name):
        matches = re.findall("-([0-9])", name)
        assert(len(matches) == 1)
        return int(matches[0])

    def extract_activation_precision(name):
        matches = re.findall("a=([0-9]+)", name)
        if len(matches) == 0:
            return None
        return int(matches[0])

    d = []
    fp32_time = None
    with open(fpath, "r") as f:
        for line in f.readlines():
            line = line.strip()
            if line == "":
                continue
            datapoint = line.split(",")
            try:
                module,task,method,W1,A1,score,time = datapoint
            except:
                continue
            W1,A1 = tuple([int(x) for x in [W1,A1]])
            if method == "cutlass" and W1 == 32:
                assert(W1 == 32)
                assert(A1 == 32)
                fp32_time = float(time)
            score,time = float(score), float(time)
            d.append([method,W1,A1,score,time])
    dd = []
    for d_point in d:
        method,_,_,score,time = d_point
        dd.append([method,score,fp32_time/time])
    print(dd)
    return dd
    

def plot_data(variable_data):
    plt.cla()
    pbatch_data = [x for x in variable_data if x[0] == "pbatch"]
    cutlass_data = [x for x in variable_data if x[0] == "cutlass"]

    pbatch_metric_and_speedups = [(x[1],x[2]) for x in pbatch_data]
    cutlass_metric_and_speedups = [(x[1],x[2]) for x in cutlass_data]

    pbatch_metric_and_speedups = sorted(pbatch_metric_and_speedups, key=lambda x:x[1])
    cutlass_metric_and_speedups = sorted(cutlass_metric_and_speedups, key=lambda x:x[1])

    pbatch_metric_and_speedups = get_pareto_points(pbatch_metric_and_speedups)
    cutlass_metric_and_speedups = get_pareto_points(cutlass_metric_and_speedups)

    print(pbatch_metric_and_speedups)
    print(cutlass_metric_and_speedups)

    plt.plot([x[0] for x in pbatch_metric_and_speedups], 
             [x[1] for x in pbatch_metric_and_speedups], 
             label="PBatch", marker="o", linewidth=5, markersize=10, color="blue", linestyle="-")
    plt.plot([x[0] for x in cutlass_metric_and_speedups],
             [x[1] for x in cutlass_metric_and_speedups],
             label="Cutlass", marker="o", linewidth=5, markersize=10, color="red", linestyle="-")

    # Adjust window
    #plt.axis([400,800,2.5,8]) 

data_path, out = sys.argv[1], sys.argv[2]
data = read_data(data_path)

plot_data(data)

plt.ylabel("End-to-End Speedup vs FP32", fontsize=20)
plt.xlabel("Average Reward", fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.tick_params(axis='both', which='minor', labelsize=15)
plt.legend(loc="best", fontsize=15)
plt.tight_layout()
plt.grid()
plt.savefig("%s.pdf" % out)
