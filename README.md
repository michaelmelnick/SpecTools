SpecTools
=========

This is a collection of objects and methods for analyzing, modelling and comparing
spectral data. It was originally written for deconvoluting UV-Visible spectral
data for semiconductor quantum dots into component Gaussians, but has been
generalized for other x-y datasets.

There are three primary objects

1) feature

Features generate output values from a mathematical function and a set of 
parameters if given output values. A collection of default functions are
provided, but users can also define their own function and parameters.

2) spectrum

This holds x-y data and a collection of features, as well as a set of methods
for least squares refinement.

3) reaction

A reaction is a container for a collection of spectra. It contains methods
for extracting data and visualizing the results.

Installation:

Download and extract the archive anywhere on your system.

In a terminal enter:

python setup.py install

Alternatively add the directory you extracted the archive to you python path. 
