# ANN model for compact modeling of Thin-TFET

## Model Overview

These models use feed forward artificial neuron network (ANN) to model emerging electronic device. This ANN model paradigm uses a special designed loss function to achieve much better accuracy in the sub-threshold region than standard quadratic loss function. Besides this, these ANN models are easy to use and guarantee to be infinitely differentiable everywhere.

We have trained two different types of model: one using logistic sigmoid functions as activation functions, and the other using hyperbolic tangent functions as activation functions. Brief descriptions are followed:

### Model with logistic sigmoid function (*sig*)

This model has only one hidden layer, which has 5 to 9 neurons in different versions. The first and third quadrants are modeled independently, then add them together.

- **Advantage**: excellent accuracy in the *deep* sub-threshold region (i.e. Vtg < 50 mV)
- **Disadvantage**: nonlinear turn-on around Vds = 0
- **When to use**: when deep sub-threshold region is important

The *sig* model comes with different versions for both first and third quadrants:

- For the 1st quadrant, there are two versions:
	1. *sig_1q_7*: which has 7 neurons in the hidden layer
	2. *sig_1q_9*: which has 9 neurons in the hidden layer
- For the 3nd quadrant, there are three versions:
	1. *sig_3q_5*: which has 5 neurons in the hidden layer
	2. *sig_3q_7*: which has 7 neurons in the hidden layer
	3. *sig_3q_9*: which has 9 neurons in the hidden layer

### Model with hyperbolic tangent functions (*tan*)

This model uses hyperbolic tangent functions. Unlike *sig* model, *tan* model use a single ANN to model both first and third quadrants.

- **Advantage**: compared to *sig* model, this model has no unintentional nonlinear turn-on around Vds = 0
- **Disadvantage**: current densities oscillate around 0 in the deep sub-threshold (Vgs < 50 mV)
- **When to use**: when the linearity of Id-Vds around Vds = 0 is important

The *tan* model comes with three different versions:

1. *tan_2h*: which has 2 hidden layers, each has 9 neurons
2. *tan_3h*: which has 3 hidden layers, each has 9 neurons
3. *tan_4h*: which has 4 hidden layers, each has 9 neurons

