import scipy.stats as stats
import sys
import numpy as np
import tqdm
from sklearn.utils.extmath import softmax
import h5py
import matplotlib.pyplot as plt
from itertools import permutations
data=['Kommission', 'Kommission@@', 'Kommissions@@', 'Kommiss@@', 'Kommissionspräsi@@',
      'Kommissionspräsident', 'Rat', 'Parlament', 'Kommissionspräsidenten',
      'Kommissionsvorschlag', 'Kommissionsmitgli@@', 'Kommissionsmitglieder',
      'Kommissionsmitglied', 'Kommissar', 'Kommissarin', 'Berichterstatterin',
      'Mitgliedstaaten', 'Parlaments', 'Vorschlag', 'Berichterstatters']



sample_width = len(data[0])

x = np.arange(sample_width)


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

elif mode == 'linear':
    k = 0
    b = 1.0
    offset = 0
    sample_width = 0
    softmax_position = 0

figsize = 20, 10
figure, ax = plt.subplots(figsize=figsize)

y_sample = distribution_func.pdf(x)
print(y_sample)
y_sample = softmax(np.expand_dims(y_sample,0)).squeeze(0)
y_sample = y_sample[:10]
print(len(y_sample))
plt.plot(data[:len(y_sample)], y_sample, marker="o")
plt.show()

"""y_sample=[[2,2,2,2,2.0,0,0,0,0,0]]
y_sample=softmax(y_sample)
plt.plot(data[:len(y_sample[0])], y_sample[0], marker="o")
print(data)
plt.show()"""

print(y_sample)