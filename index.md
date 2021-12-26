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

Cells are the basic structural and functional units of life. Cells can be divided in two main types: **eukaryotic**, which contain a nucleus, and prokaryotic cells, which do not have a nucleus, but a nucleoid region is still present. The **prokaryotic** cells are simpler and smaller than eukaryotic cells and characterize mainly bacteria and archaea, two of the three domains of life. On the other hand, the eukaryotic cells are wider and more complex than prokaryotic ones and are present in plants, animals, fungi, protozoa, and algae. The eukaryotic cells are different in types, sizes, and shapes. However, for descriptive purposes, the concept of a **generalized cell** is introduced. A cell consists of three parts:
* The **cell membrane** which separates the material inside the cell from the material outside the cell. It maintains the integrity of a cell and controls passage of materials into and out of the cell.
* The **nucleus** which is formed by a nuclear membrane around a fluid nucleoplasm, is the control center of the cell. It contains the deoxyribonucleic acid (DNA), the genetic material of the cell.
* The **cytoplasm**, a gel-like fluid inside the cell. It is the medium for chemical reaction and contains the **organelles** each of which has a specific role in the function of the cell.


Cell cycle is the most fundamental biological process underlying the existence and propagation of life in time and space. The cell cycle is a 4-stage process consisting of Gap 1 (G1), synthesis (S), Gap 2 (G2) and mitosis (M), which a cell undergoes as it grows and divides. After completing the cycle, the cell either starts the process again from G1 or exits the cycle through G0. From G0, the cell can undergo terminal differentiation.

![image](assets/img/Cell_Cycle.png)
