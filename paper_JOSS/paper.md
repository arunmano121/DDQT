---
title: 'Defect Detection and Quantification Toolbox (DDQT)'
tags:
  - Python
  - Nondestructive Testing
  - Structural Health Monitoring
  - Defect Detection
  - Defect Quantification
  - Outlier Detection
  - Mahalanobis Distance
  - Isolation Forest
  - Principal Component Analysis
authors:
  - name: Arun Manohar
    affiliation: 1
    orcid: 0000-0003-2446-5008
affiliations:
  - name: Department of Structural Engineering, University of California,
    San Diego, CA 92037, USA
    index: 1
date: 14 April 2021
bibliography: paper.bib
---

# Summary

The domain of `Nondestructive Testing (NDT)` [@cartz] and `Structural Health
Monitoring (SHM)` [@farrar] comprise of techniques that are used to evaluate
the state of health of a structure without causing any damage to the
structure being inspected. Typical examples of structures being inspected
include aircraft components, bridges, nuclear reactors, etc. 

`Defect Detection and Quantification Toolbox (DDQT)` aims to provide a
framework to automate the routine tasks for researchers in the field of `NDT`
and `SHM`.


# Statement of Need

In the fields of `NDT` and `SHM`, experiments are performed using a variety of
modalities like Ultrasound, Infrared Thermography, X-Rays, etc. [@bray]
`Matlab` is often used to perform experiments. Typically, the resulting data
is very often a 3D dataset comprising of time axis and 2D spatial axis. Once
the data is obtained, a variety of visualization checks are performed.
Following this, features are created at each and every spatial location in
the time series. Some examples of the types of features are based on time
series, gradients, spatial filters, etc. At this stage, to avoid the `curse of
dimensionality` [@curse], a feature reduction step is often performed using
`Greedy feature selection` [@greedy] in the `supervised defect detection`
case [@supervised], or `PCA` [@pca] in case of `unsupervised defect
detection` [@unsupervised]. Once the subset of meaningful features is
identified, the defect regions are detected using a statistical distance
metric like `Mahalanobis distance` [@mahalanobis]. In `DDQT`, we also propose
a new way to identify defects using `Isolation Forests` [@isolation].
Following this, the performance of the feature space and detection algorithms
is quantified using `Receiver Operating Curves (ROC)` curves compared to the
raw data. The performance is quantified using commonly used classification
metrics such as `True Positive Rate (TPR)`, `False Positive Rate (FPR)` and
`Area Under Curve (AUC)` [@metrics]. A visual representation of these metrics
is presented using charts to aid the user.

While there are clearly defined steps that researchers in field of `NDT` and
`SHM` often pursue, to the best of our knowledge there are no offerings that
provide a framework so researchers can mainly focus on the feature space and
tweaking the defect detection algorithms. With very minimal edits to the
driver program, researchers can visualize the results with ease and focus on
the underlying `physics` to improve the performance of defect detection
instead of spending much time and effort on setting up the pipeline process.
`DDQT` aims to bridge this gap. 


# Getting started

A detailed description on how to get started using `DDQT` is shown at [DDQT's
`readthedocs` page](https://ddqt.readthedocs.io/en/latest/getting_started.html).

Code reference is shown at [DDQT's `modules` page](https://ddqt.readthedocs.io/en/latest/modules.html).


# References
