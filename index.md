---
layout: page
title: Cell cycle prediction
subtitle:  An unsupervised approach
cover-img: assets/img/Cell-cycle-cover-img.jpg
thumbnail-img: assets/img/Cell-cycle-cover-img.jpg
share-img: assets/img/Cell-cycle-cover-img.jpg
use-site-title: true
---


# Abstract

Advances in omics technologies make it possible to study cellular dynamics, providing accurate information on which genes encoded in our DNA are turned on or off in the continuously changing transcriptome. In particular, RNA-seq provides a powerful means to analyze molecular mechanisms underlying cell-state transitions, leading to an unprecedented opportunity to reveal latent biological processes. As a result, the reconstruction of cell development processes from RNA sequences has attracted much attention in recent years. Still, it remains a challenge due to the heterogeneous nature of the processes. The underlying idea in most methods proposed is that there is a biological process responsible for the main variations in the data. Then the goal is to infer the trajectory of that process in the gene expression space so that its effects can be removed. It allows the delineation of other cell subpopulations, which can be crucial to studying tumor evolution. This project explores computational techniques for pseudo-time inference of the cell cycle process from RNA sequences. This study presents different unsupervised approaches to this problem: an autoencoder approach (), a Generetive adversarial learning approach () and a Bayesian Gaussian process latent variable model based approach ().

# Introduction
------------------------------------------------------

Cells are the basic structural and functional units of life. Cells can be divided in two main types: **eukaryotic**, which contain a nucleus, and prokaryotic cells, which do not have a nucleus, but a nucleoid region is still present. The **prokaryotic** cells are simpler and smaller than eukaryotic cells and characterize mainly bacteria and archaea, two of the three domains of life. On the other hand, the eukaryotic cells are wider and more complex than prokaryotic ones and are present in plants, animals, fungi, protozoa, and algae. The eukaryotic cells are different in types, sizes, and shapes. However, for descriptive purposes, the concept of a **generalized cell** is introduced. A cell consists of three parts:
* The **cell membrane** which separates the material inside the cell from the material outside the cell. It maintains the integrity of a cell and controls passage of materials into and out of the cell.
* The **nucleus** which is formed by a nuclear membrane around a fluid nucleoplasm, is the control center of the cell. It contains the deoxyribonucleic acid (DNA), the genetic material of the cell.
* The **cytoplasm**, a gel-like fluid inside the cell. It is the medium for chemical reaction and contains the **organelles** each of which has a specific role in the function of the cell.


Cell cycle is the most fundamental biological process underlying the existence and propagation of life in time and space. The cell cycle is a 4-stage process consisting of Gap 1 (G1), synthesis (S), Gap 2 (G2) and mitosis (M), which a cell undergoes as it grows and divides. After completing the cycle, the cell either starts the process again from G1 or exits the cycle through G0, a state of quiescence.

| ![cell_cycle](assets/img/Cell_Cycle.png) | 
|:--:| 
| *cell cycle* |


* **G0** is a resting phase. In this phase, the cell has left the cycle and has stopped dividing. The cell cycle always starts with this phase. Cells in multicellular eukaryotes generally enter the quiescent G0 state from G1 and may remain quiescent for long periods of time, possibly indefinitely.
* **G1** is the first phase of the interphase (the phases between two M phases). G1 indicates the end of the previous M phase until the beginning of DNA synthesis, it is often also called the growth phase. During G1 phase, the biosynthetic activities of the cell resume at a high rate. The cell increases its supply of proteins, increases the number of organelles, and grows in size. At the end of G1 phase, a cell has three options. To continue cell cycle and enter S phase or to stop cell cycle and enter G0 phase for undergoing differentiation.
* **S** phase characterized by DNA replication and protein synthesis as well as a rapid cell growth. 
* **G2** phase in which the cell checks if there is any DNA damage within the chromosomes. In case of any anomaly, the cell will repair the DNA or trigger the apoptosis of the cell.
* **M** (mitotic phase) which consists of nuclear division. Mitosis is the process by which a eukaryotic cell separates itself into two identical daughter cells.


# Dataset description and preprocessing
---------------------------------------------------------------


|            | cell 1 | cell 2 | cell 3 | cell 4 | ... |
|------------|--------|--------|--------|--------|-----|
| total_UMIs | 37509  | 34809  | 85843  | 48921  | ... |
| count 1    | 0      | 0      | 0      | 0      | ... |
| count 2    | 0      | 0      | 0      | 0      | ... |
| count 3    | 0      | 0      | 0      | 0      | ... |
| ...        | ...    | ...    | ...    | ...    | ... |

Table: *First rows and columns of CHLA9 dataset*

The dataset used in our analysis is CHLA9.loom, containing information about more than 5000 different cells. Notice that the initial format of our data is a loom format, designed to efficiently hold large omics datasets. As shown in the table above, each column of the dataset describes a different cell and it is named and identified by a unique string, part of which corresponds to the DNA sequence of the cell. For each column of the dataset (for each cell) we have more than 60000 positive natural numbers associated to it. Each of these numbers represents a specific count of genes inside the cell. Finally, for each cell, we also have an attribute called "TotalUMIs" (one of the counts) representing the sum of the aforementioned counts. This number is often introduced in omic datasets since most of the gene counts are zero (our dataset is sparse).

| ![dataset_sparsity](assets/img/Dataset_sparsity.png) | 
|:--:| 
| *dataset sparsity* |


Each row of our dataset is indexed by a list of keys. Among them, we decided to filter our initial dataset so that it contains only the counts of the genes whose genetype is 'protein_encoding' since, according to literature, it is generally easier to predict the phase of their cycle and since we have a huge number of these genes.
Moreover, we filtered the dataset using the attributed "TotalUmis", deleting all the cells with a number of counts outside the 25-75% interquantile range. Here below we show the boxplot used to filter the dataset and the genetype distribution.

<iframe src="assets/plots/subplots_dataset_preprocessing.html" width="100%" height="800"> </iframe>

The reason why we filtered the cells with a too small or too large number of genes is due to the fact that the phase of those cell could be more difficult to determine when the number of these observed genes does not lie inside the 25-75%. Indeed, if the count was outside this range, we could have too few or too many genes to take into account in our future models and therefore we could easily underfit or overfit the data.
Moreover, we removed from our dataset the half of the genes of our dataset (half of the rows) that had the bigger number of zeros inside them because, with the same argument as before, we would like to focus on the genes which are sufficiently observed in our dataset. 
Finally, as suggested by our supervisor, we applied a standardization technique often used in omic domain. For each cell, we divided each count of genes (each column of our dataframe) by the total number of the counts of the genes inside the cell. In this way we now have to deal with real values in the interval [0,1] instead of discrete values which could cause problems when applying Machine Learning models to them. Additionally, since we had values really close to zero in the entries of our dataframe, we added to all the entries the minimum value in all the entries of the dataframe and, subsequently, we applied a log transformation elementwise. This standardizing procedure will allow us to work with real values sufficiently far from zero, allowing us to avoid numerical issues.

# Task and methods description
---------------------------------------------------------------

The goal of our project will be to determine the phase of the cells using the described dataset. To this end, we will use different unsupervised machine learning techniques:
* Autoencoders
* Autoencoders with residual networks
* Gaussian process latent variable model (GPLVM)
We will now describe how this methods work and the obtained results on our specific case.

# Autoencoders

The objective is to infer pseudo-time/embedding $\mathbf{X}_n$ for cell $n$ from its transcriptome profile $\mathbf{y}_n$ a column vector containing the expression
levels of G genes. To do that we search for a transformation $\mathbf{x}_{n}=\mathcal{F}\left(\mathbf{y}_{n}\right)=\mathbf{W} \mathbf{y}_{n}$ and an inverse transformation $\hat{\mathbf{y}}_{n}=\mathcal{F}^{-1}\left(\mathbf{x}_{n}\right)=\mathbf{W}^{\mathrm{T}} \mathbf{x}_{n}$, such that the total error $\sum_{n=1}^{N}||\mathbf{y}_{n}-\hat{\mathbf{y}}_{n}||^{2}$ is minimized. In particular we are looking for transformation $\mathcal{F}^{-1}(\cdot)$ and $\mathcal{F}(\cdot)$ such that they are nonlinear periodic functions and therefore sensitive to circular trajectories.
To extrapolate this non-linear transformation we applied a machine learning technique called Autoencoder, specifically an asymmetric Autoencoder. In the encoder, we use a standard multi-layer perceptron with hyperbolic tangent activation functions, in the decoder we use cosine and sine as the activation functions in the first layer, followed by a second layer performing linear transformations. We use the least square error as the optimization target with L2 regularization, formally:
$$
\underset{\mathbf{w}_{i}, \mathbf{v}}{\operatorname{argmin}} \sum_{n=1}^{N}||\mathbf{y}_{n}-\hat{\mathbf{y}}_{n}||_{2}^{2}+\sum_{i} \alpha_{i}||\mathbf{W}_{i}||_{L}^{2}+\beta||\mathbf{V}||_{L}^{2}
$$
The network is implemented using Keras with TensorFlow, which optimizes the parameters using gradient descent.
