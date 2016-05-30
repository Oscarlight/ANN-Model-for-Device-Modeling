## ANN compact model by Mingda Li

from ffnn import *
# --------------------------------------- 
#   	       sig model
# The first argument is the 1st quadrant model,
# which can be: 'sig_1q_7/'
# and second argument is the 3nd quadrant model
# which can be: 'sig_3q_5/', 'sig_3q_7/'
# ---------------------------------------
sig_ann = make_sig('sig_1q_7/', 'sig_3q_5/', 2)
# --------------------------------------- 
#   	       tan model
# For different models:
# a) tan_ann = make_tan('tan_2h/', 3)
# b) tan_ann = make_tan('tan_3h/', 4)
# c) tan_ann = make_tan('tan_4h/', 5)
# ---------------------------------------
tan_ann = make_tan('tan_2h/', 3)

# plot
x_list = np.linspace(-0.1, 0.4, 100)
c_list = [i*0.01 for i in range(0, 41)]
sig_save = plot(sig_ann, x_list, c_list, TYPE = 'f', LOG = False)
tan_save = plot(tan_ann, x_list, c_list, TYPE = 'f', LOG = False)

# save (an example)
# saveData("family", "tan", c_list, x_list, tan_save, 1, "Vtg")
