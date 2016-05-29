# ANN model for compact modeling of ThinTFET
## Model Overview:
There are many models for you to choose, they can be seperated into
groups, both of them are infinitely differentiable:
### tan model
1. **Description:** in this model, all activation function are tanh.
2. **Advantage:** this model has better linearity around Vds = 0
3. **Disadvantage:** current density can oscillate around 0 when Vgs < 50 mV
4. **Verdict:** use it if the linearity of Id-Vds around Vds = 0 is important
5. **Versions:** 
	there are three versions of this model provided:
    
    1. tan_2h: with two hidden layers
    2. tan_3h: with three hidden layers
    3. tan_4h: with four hidden layers
    
    A model with more hidden layers provides _slightly_ better accuracy. 
	The smallest model, i.e. tan_2h may be good to start with. 

### sig model
1. **Description:** in this model, all activation function are sigmoid, and the first and third quadrant are modelled seperately then add together
2. **Advantage:** excellent accuracy in the _deep_ sub-threshold region (i.e. Vtg < 50 mV)
3. **Disadvantage:** nonlinear turn-on around Vds = 0
4. **Verdict:** use it if accuracy in deep subthreshold region is important
5. **Versions:** 
    - for the 1st quadrant, there are two versions:
        1. sig_1q_7: with 7 neurons in the hidden layer
        2. sig_1q_9: with 9 neurons in the hidden layer
    - for the 3nd quadrant, there are three versions:
  	1. sig_3q_5: with 5 neurons in the hidden layer
 	2. sig_3q_7: with 7 neurons in the hidden layer
	3. sig_3q_9: with 9 neurons in the hidden layer


