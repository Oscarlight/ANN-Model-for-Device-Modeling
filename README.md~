# ANN model for compact modeling of ThinTFET
    There are many models for you to choose, they can be seperated into
    groups, both of them are infinitely differentiable:

1. tan model
    - ** Description: ** in this model, all activation function are tanh.
    - ** Advantage: ** this model has better linearity around Vds = 0
    - ** Disadvantage: ** current density can oscillate around 0 when Vgs < 50 mV
    - ** Verdict: ** use it if the linearity of Id-Vds around Vds = 0 is important
    - ** Versions: ** there are three versions of this model provided:
         1. tan_2h: with two hidden layers
	 2. tan_3h: with three hidden layers
	 3. tan_4h: with four hidden layers
	more hidden layers provide slightly better accuracy, 
	the smallest model, i.e. tan_2h may be good to start with. 

2. sig model
    - ** Description: ** in this model, all activation function are sigmoid, and the first and third quadrant are modelled seperately then add together
    - ** Advantage: ** excellent accuracy in the _deep_ sub-threshold region (i.e. Vtg < 50 mV)
    - ** Disadvantage: ** nonlinear turn-on around Vds = 0
    - ** Verdict: ** use it if accuracy in deep subthreshold region is important
    - ** Versions: ** 
	 for the 1st quadrant, there are two versions:
	 1. sig_1q_7: with 7 neurons in the hidden layer
         2. sig_1q_9: with 9 neurons in the hidden layer
  	 for the 3nd quadrant, there are three versions:
  	 1. sig_3q_5: with 5 neurons in the hidden layer
 	 2. sig_3q_7: with 7 neurons in the hidden layer
	 3. sig_3q_9: with 9 neurons in the hidden layer

