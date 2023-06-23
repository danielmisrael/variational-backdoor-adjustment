# variational-backdoor-adjustment


This repository contains the implementation of Variational Backdoor Adjustment (VBA) in the following paper:

>__Escaping the Curse of High Dimensional Backdoor Adjustment__
>
> Authors: Daniel Israel, Guy Van den Broeck, Aditya Grover (link to be added)
>
>Abstract: Backdoor adjustment is an important technique in causal inference for estimating interventional quantities from purely observational data. In medical settings for example, backdoor adjustment can be used to control for confounding and isolate the effectiveness of a treatment. However, high dimensional treatments and confounders pose a series of potential pitfalls. Backdoor adjustment over high dimensional confounders is often intractable. As a remedy, previous approaches often model confounding with latent variables in VAEs, but these methods perform backdoor adjustment over unidentifiable, unobserved variables, leading to inconsistent estimates. In this work, we show that a generative modeling approach can be applied to backdoor adjustment in a fully identified high dimensional setting. Specifically, we cast backdoor adjustment as an optimization problem in variational inference that is constrained by the distribution of a fully observed confounder. Empirically, our method is able to estimate interventional likelihood in a variety of high dimensional settings, including semi-synthetic X-ray medical data. To the best of our knowledge, this is the first application of backdoor adjustment in which all the relevant variables are high dimensional.

## Environment Setup
1. Clone the repository.
2. Install the necessary packages with
```
conda env create -f environment.yml
conda activate variationalbackdoor
pip install -e .
```
3. To run the code involving autoregressive flow, clone the [FFJORD repo](https://github.com/rtqichen/ffjord) and follow the instructions. Please note that these models are more expensive to train than the vanilla neural networks and VAEs used in the experiments.

## Usage
A documented example of VBA is given in ```example.py```. Adapting the code for real-world data will likely require a different parameterization of models.

The experiments are divided into ```linear_gaussian```, ```mnist```, and ```xray```. 

### Linear Gaussian
1. Run ```train_lg.py```
2. For table values, run ```generate_lg_table.py```, then ```load_lg_table.py```
3. For plots, run ```lg_optimization_plot.py``` and ```lg_sampling_plot.py```

### MNIST
1. Run ```train_mnist_vae.py``` and ```train_mnist_vb.py```
2. For plots, run ```mnist_plot.py```

### X-Ray
1. Run ```train_xray.py``` (Warning: FFJORD must be properly configured and trained separately)
2. For table values, run ```generate_xray_table.py```, then ```load_xray_table.py```
3. For plots, run ```generate_xray_plot.py```, then ```load_xray_plot.py```
   


