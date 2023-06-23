import pickle 
import numpy as np

def print_results(name, dis):
    max_seed = 5
    with open(f'results/lg/{name}_{dis}_seed=0.pickle', 'rb') as handle:
        d = pickle.load(handle)
    results = {}
    for key in d:
        results[key] = []


    for i in range(max_seed):
        with open(f'results/lg/{name}_{dis}_seed={i}.pickle', 'rb') as handle:
            d = pickle.load(handle)

        for key in d:
            results[key].append(d[key])

    for key in d:
        arr = np.array(results[key])
        print(key, 'mean', np.mean(arr))
        print(key, 'stderr', np.std(arr) / np.sqrt(max_seed))


print('ID')
print('Separate training')
print_results('sep', 'id')
print('Finetuning')
print_results('finetuned', 'id')
print('OD')
print_results('sep', 'od')
print('Finetuning')
print_results('finetuned', 'od')


