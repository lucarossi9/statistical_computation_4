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

Cell cycle is the most fundamental biological process underlying the existence and propagation of life in time and space. The cell cycle is a 4-stage process consisting of Gap 1 (G1), synthesis (S), Gap 2 (G2) and mitosis (M), which a cell undergoes as it grows and divides. After completing the cycle, the cell either starts the process again from G1 or exits the cycle through G0. From G0, the cell can undergo terminal differentiation.

![image](assets/img/Cell_Cycle.png)
