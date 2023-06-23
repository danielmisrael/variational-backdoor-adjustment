import pickle
import scipy.stats as st
import matplotlib.pyplot as plt

def confidence_interval(data, confidence=0.90):
    return st.norm.interval(confidence, loc=data.mean(), scale=st.sem(data))

def process_dict(dict):
    mean = []
    lower = []
    upper = []
    for key in dict:
        bpd = dict[key]
        print(bpd)
        mean.append(bpd.mean())
        interval = confidence_interval(bpd)
        lower.append(interval[0])
        upper.append(interval[1])

    return mean, lower, upper


with open('results/xray/xray_no_joint_accuracies_bpd.pickle', 'rb') as handle:
    sep = pickle.load(handle)


with open('results/xray/xray_finetuned_accuracies_bpd.pickle', 'rb') as handle:
    finetuned = pickle.load(handle)


plt.rcParams.update({'font.size': 15})

acc = [0, 0.3, 0.5, 0.7, 1]

mean, lower, upper = process_dict(finetuned)
plt.plot(acc, mean, label='Finetuning', color='royalblue')
plt.fill_between(acc, lower, upper, alpha=.5, color='royalblue', linewidth=0)

mean, lower, upper = process_dict(sep)
plt.plot(acc, mean, label='Separate Training', color='limegreen')
plt.fill_between(acc, lower, upper, alpha=.5, color='limegreen', linewidth=0)



plt.ylim((4.5,7))
plt.ylabel('Bits per Dimension')
plt.xlabel('Treatment Accuracy')
plt.legend()
plt.show()



