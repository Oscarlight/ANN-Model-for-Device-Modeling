# ANN model for compact modeling of Thin-TFET

## 1. Model Overview

These models use feed forward artificial neuron network (ANN) to model emerging electronic device. This ANN model paradigm uses a special designed loss function to achieve much better accuracy in the sub-threshold region than standard quadratic loss function. Besides this, these ANN models are easy to use and guarantee to be infinitely differentiable everywhere.

We have trained two different types of model: one using logistic sigmoid functions as activation functions, and the other using hyperbolic tangent functions as activation functions. Brief descriptions are followed:

### a) Model with logistic sigmoid function (*sig* model)

This model has only one hidden layer, which has 5 to 9 neurons in different versions. The first and third quadrants are modeled independently, then add them together.

- **Advantage**: excellent accuracy in the *deep* sub-threshold region (i.e. Vtg < 50 mV)
- **Disadvantage**: nonlinear turn-on around Vds = 0
- **When to use**: when deep sub-threshold region is important, also *sig* model is light-weighted compared to *tan* model. If you can live with the nonlinearity around Vds = 0. *sig* model is your friend.

The *sig* model comes with different versions for both first and third quadrants:

- For the 1st quadrant, there are one versions:
	1. *sig_1q_7*: which has 7 neurons in the hidden layer
- For the 3nd quadrant, there are two versions:
	1. *sig_3q_5*: which has 5 neurons in the hidden layer
	2. *sig_3q_7*: which has 7 neurons in the hidden layer

Recommend to use *sig_1q_7* and *sig_3q_5*.

### b) Model with hyperbolic tangent functions (*tan* model)

This model uses hyperbolic tangent functions. Unlike *sig* model, *tan* model use a single ANN to model both first and third quadrants.

- **Advantage**: compared to *sig* model, this model has no unintentional nonlinear turn-on around Vds = 0
- **Disadvantage**: current densities oscillate around 0 in the deep sub-threshold (Vgs < 50 mV)
- **When to use**: when the linearity of Id-Vds around Vds = 0 is important

The *tan* model comes with three different versions:

1. *tan_2h*: which has 2 hidden layers, each has 9 neurons
2. *tan_3h*: which has 3 hidden layers, each has 9 neurons
3. *tan_4h*: which has 4 hidden layers, each has 9 neurons

Recommend to start with *tan_2h*, the simplest version yet almost as accurate as its bigger brothers (i.e. *tan_3h* and *tan_4h*)

## 2. Use the model (in Python)

The interfaces to use this model are in *neuralFET.py*. Here is a short tutorial of how to use both *sig* model and *tan* model. The main goal we would like to achieve is: for given **Vgs** and **Vds**, get current density **Id**.

### a) Use *sig* model

First, read the model file, and create a function accordingly. For example, if we would like to use *sig_1q_9* for the first quadrant, and *sig_3q_7* for the thrid quadrant, then:
```python
sig_ann = make_sig('sig_1q_9/', 'sig_3q_7/', 2)	
```
The last *"2"* indicate there is *n-1* hidden layer(s).
Now you can get current density from Vgs and Vds by:
```python
id = sig_ann(Vgs, Vds) 
# Vgs and Vds have unit of V, id has unit of A/m
```

### a) Use *tan* model
Just like *sig* model, if we would like to use *tan_2h* version:
```python
tan_ann = make_tan('tan_2h/', 3)	
```
Then to get current density, simple use:
```python
id = tan_ann(Vgs, Vds)
# Vgs and Vds have unit of V, id has unit of A/m 	
```
-------
There is a *plot* function can be very handy to plot out the whole model at once, the signature of this function is in *ffnn.py*:
```python
data = plot(current, x_list, c_list, TYPE = "t", LOG = False)
```
**Current** is the model function, such as sig_ann or tan_ann; 
For family plot, **x_list** is a list of Vds and **c_list** is a list of Vgs; 
For transfer plot, **x_list** is a list of Vgs and **c_list** is a list of Vds;
**TYPE = "t"** for transfer plot, **TYPE = "f"** for family plot;
**LOG = False** means linear plot, **LOG = True** means semi-log plot.
*plot* function returns all the plotted data points in a 2D list.

To save this 2D list into csv file, there is also a *saveData* function to help do so. This function is in *ffnn.py*.

Enjoy using this model! 
