# Physics-guided Neural Networks (PGNN) : An Application In Lake Temperature Modelling

This repository provides code for the PGNN paper. If you're using the code in your research please cite the paper https://arxiv.org/abs/1710.11431.

## Abstract:
This paper introduces a novel framework for combining scientific knowledge of physics-based models with neural networks to advance scientific discovery. This framework, termed as physics-guided neural network (PGNN), leverages the output of physics-based model simulations along with observational features to generate predictions using a neural network architecture. Further, this paper presents a novel framework for using physics-based loss functions in the learning objective of neural networks, to ensure that the model predictions not only show lower errors on the training set but are also scientifically consistent with the known physics on the unlabeled set. We illustrate the effectiveness of PGNN for the problem of lake temperature modeling, where physical relationships between the temperature, density, and depth of water are used to design a physics-based loss function. By using scientific knowledge to guide the construction and learning of neural networks, we are able to show that the proposed framework ensures better generalizability as well as scientific consistency of results.

## Datasets :
This paper considers the following two example lakes to demonstrate the effectiveness of PGNN framework.
1. Lake Mille Lacs in Minnesota, USA
2. Lake Mendota in Wisconsin, USA

Please note that the paper provides an semi-supervised approach framework, where the mean squared error on the temperature predictions are computed using a labeled dataset whereas the physics based loss can be computed from an unlabeled dataset. The labelled and unlabeled datasets can be found in the 'datasets\\' directory under the name '[lake].mat' and '[lake]\_sampled.mat' respectively. [lake] should be replaced by 'mendota' for Lake Mendota and 'mille_lacs' for Lake Mille Lacs.


### Dependencies :

* Python 3.7.3
* Keras 2.2.5
* Tensorflow 1.14.0

## Using the code :

The repository contains code and datasets needed for training and testing the PGNN framework described in the paper.

1. To save the models and the results after training please create a '\results\\' directory. 
```
mkdir results
```
2. Then run the script '\models\PGNN.py'. 
```
cd models
python PGNN.py
```
The hyperparameters and the datasets for the PGNN framework can be changed from the script.

Note: An alternative version of the PGNN script with parser options can be used too. 
Example for Lake Mille Lacs with Adam Optimizer (other hyperparameters can be tuned accordingly):
```
cd hybrid
python hpd.py --dataset mille_lacs --optimizer_val 2 --data_dir ../datasets/ --use_YPhy 1 --lambda 100.0
```


### Using the code for Hybrid Modeling:

The **HPD model** is same as the PGNN0, i.e., the physics-loss is not used $\lambda_{PHY}=0$. To run the HPD model use the following script '\hybrid\hpd.py'
Example for Lake Mille Lacs with Adam Optimizer:
```
cd hybrid
python hpd.py --dataset mille_lacs --optimizer_val 2 --data_dir ../datasets/ --use_YPhy 1 --lambda 0.0
```

To use the **Residual (Res) model**, without the use of the addition YPhy input use the script '\hybrid\res_nn.py'
Example for Lake Mille Lacs with Adam Optimizer:
```
cd hybrid
python res_nn.py --dataset mille_lacs --optimizer_val 2 --data_dir ../datasets/ --use_YPhy 0 --lambda 0.0
```

To run the **Hybrid-Residual (HPD-Res) model**, use the following:
```
cd hybrid
python res_nn.py --dataset mille_lacs --optimizer_val 2 --data_dir ../datasets/ --use_YPhy 1 --lambda 0.0
```
