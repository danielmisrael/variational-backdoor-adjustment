import pickle 
import numpy as np

with open('results/xray/sep_table_results.pickle', 'rb') as handle:
    d = pickle.load(handle)

print('Separate')
for key in d:
    print(key)
    arr = np.array(d[key])
    print('mean', np.mean(arr))
    print('stderr', np.std(arr) / np.sqrt(len(arr)) )


with open('results/xray/finetuned_table_results.pickle', 'rb') as handle:
    d = pickle.load(handle)



print('Finetuned')
for key in d:
    print(key)
    arr = np.array(d[key])
    print('mean', np.mean(arr))
    print('stderr', np.std(arr) / np.sqrt(len(arr)) )

