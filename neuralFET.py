## ANN compact model by Mingda Li


from ffnn import *

''' There are many models for you to choose, they can be seperated into
	groups, both of them are infinitely differentiable:
		1. tan model
			Description: in this model, all activation function are tanh.
			Advantage: this model has better linearity around Vds = 0
			Disadv.: current density can oscillate around 0 when Vgs < 50 mV
			Verdict: use it if the linearity of Id-Vds around Vds = 0 is 
					 important
			Versions: there are three versions of this model provided:
						a) tan_2h: with two hidden layers
						b) tan_3h: with three hidden layers
						c) tan_4h: with four hidden layers
					  more hidden layers provide slightly better accuracy, 
					  the smallest model, i.e. tan_2h may be good to start with. 
		2. sig model
			Description: in this model, all activation function are sig.
						 the first and third quadrant are modelled seperately
						 then add together
			Advantage: excellent accuracy in the deep sub-threshold region
					   (i.e. Vtg < 50 mV)
		    Disadv.: nonlinear turn-on around Vds = 0
		    Verdict: use it if accuracy in deep subthreshold region is important
		    Versions: for the 1st quadrant, there are two versions:
		    			a) sig_1q_7: with 7 neurons in the hidden layer
		    			b) sig_1q_9: with 9 neurons in the hidden layer
		    		  for the 3nd quadrant, there are three versions:
		    		    a) sig_3q_5: with 5 neurons in the hidden layer
		    			b) sig_3q_7: with 7 neurons in the hidden layer
		    			c) sig_3q_9: with 9 neurons in the hidden layer     
'''

# --------------------------------------- 
#   	       sig model
# ---------------------------------------
# sig_ann = make_sig('sig_1q_9/', 'sig_3q_9/', 2)
# --------------------------------------- 
#   	       tan model
# a) tan_ann = make_tan('tan_2h/', 3)
# b) tan_ann = make_tan('tan_3h/', 4)
# c) tan_ann = make_tan('tan_4h/', 5)
# ---------------------------------------
tan_ann = make_tan('tan_2h/', 3)

# plot
x_list = np.linspace(-0.2, 0.4, 1000)
c_list = [i*0.05 for i in range(0, 8)]
plot(tan_ann, x_list, c_list, TYPE = 'f')


# Family
# c.saveFamily(fData, vtg_list, 15e-5)
# saveData("family", fData, vtg_list, vds_list, savelist, 1, "Vtg")

# # transfer
# c.saveTransfer(fData, vds_list, 15e-5)
# saveData("transfer", fData, vds_list, vtg_list, savelist, 1, "Vds")


