import scipy.stats as stats
import sys
import numpy as np
import tqdm
from sklearn.utils.extmath import softmax
import h5py
from itertools import permutations

mode = 'gaussian'

def scatter(a, dim, index, b): # a inplace
    expanded_index = tuple([index if dim==i else np.arange(a.shape[i]).reshape([-1 if i==j else 1 for j in range(a.ndim)]) for i in range(a.ndim)])
    a[expanded_index] = b
    print("a;",a)
    print(expanded_index)

if mode == 'gaussian':
    std = 1
    offset = 0
    mean = 0
    sample_width = 0
    softmax_position = "presoftmax"
    softmax_temperature = 1
    output_path = "out.txt"

    distribution_func = stats.norm(mean, std)


# load the order information
data=[[0,1,2,3,4],[1,2,3,4,0],[2,3,4,0,1],[3,2,1,4,0],[4,3,2,1,0]]

print(len(data))
print(data)

#assert len(data) == len(data[0])

if sample_width == 0:
    sample_width = len(data[0])

x = np.arange(sample_width) + offset


if mode == 'gaussian':
    y_sample = distribution_func.pdf(x)

y_sample = y_sample / softmax_temperature
print(y_sample)
y_sample = softmax(np.expand_dims(y_sample,0)).squeeze(0)
print(sum(y_sample))


y = np.zeros(len(data[0]))

y[:sample_width] = y_sample

print(y[:sample_width])

label_weights = np.zeros((len(data[0]), len(data[0])), dtype=np.float32)

for idx in tqdm.tqdm(range(len(data[0]))):
    sort_index = np.array(data[idx])
    resort_index = np.zeros(len(data[0]), dtype=np.int)
    natural_index = np.arange(len(data[0]))
    try:
        scatter(resort_index, 0, sort_index, natural_index)
    except:
        print("resort_index:", resort_index, " natural_index:", natural_index)
        print("resort_index_len:", len(resort_index), " natural_index_len:", len(natural_index))
    weight = y[resort_index]
    label_weights[idx] = weight

f = h5py.File(output_path,'w')
print("f created")
print(h5py.File)

print(label_weights)

f.create_dataset('weights', data=label_weights)
print("f finished")
f.close()
print("done")

