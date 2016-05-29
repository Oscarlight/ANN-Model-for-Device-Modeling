import numpy as np 
import math
import operator
import matplotlib.pyplot as plt

# --------------------------------------- 
#   	       ANN Model
# ---------------------------------------
def importWB(fModel, nlayer):
	wlst = []
	blst = []
	for i in range(0, nlayer): # <-<-<-<- # of layer
		warray = np.genfromtxt(fModel+'w'+str(i)+'.csv', delimiter=',').tolist()
		barray = np.genfromtxt(fModel+'b'+str(i)+'.csv', delimiter=',').tolist()
		if isinstance(warray[0], float):
			warray = [warray]
		if isinstance(barray, float):
			barray = [barray]
		wlst.append(warray)
		blst.append(barray)
	return (wlst, blst)

def sigmoid(x) :
	try:
		y = math.exp(-x)
	except OverflowError:
		y = float('inf')
	return 1.0 / (1.0 + y)

def gemv(weight, x, bias):
	act = lambda w, b: sum(map(operator.mul, w, x)) + b
	return map(act, weight, bias)

def forwardprop(fun, w, b, x):
	if not b:
		return x
	else:
		x = map(fun, gemv(w[0], x, b[0]))
		return forwardprop(fun, w[1:], b[1:], x)

## Get current density (uA/um) from Vds, Vgs
## (i.e. the functions you need)
def make_sig(fModel1, fModel2, nlyr):
	''' for all sigmoid activation func. model
		fModel1: dir of the 1st quadrant model
		fModel2: dir of the 3rd quadrant model '''
	wb1 = importWB(fModel1, nlyr)
	wb2 = importWB(fModel2, nlyr)
	def current(Vg, Vd):
		v1 = [(Vd-0.2)*5, (Vg-0.2)*5]
		v2 = [(Vd+0.05)*20, (Vg-0.2)*5]
		i1 = (forwardprop(sigmoid, wb1[0], wb1[1], v1)[0]) * 275
		i2 = (forwardprop(sigmoid, wb1[0], wb2[1], v2)[0]) * 45
		return i1 + i2
	return current

def make_tan(fModel, nlyr):
	''' for all tanh activation func. model '''
	wb = importWB(fModel, nlyr)
	def current(Vg, Vd):
		v = [(Vd-0.15)*4, (Vg-0.2)*5]
		i = (forwardprop(math.tanh, wb[0], wb[1], v)[0]) * 275
		return i
	return current


# --------------------------------------- 
#   	     plot and save
# ---------------------------------------

def plot(current, x_list, c_list, TYPE = "t", LOG = False):
	savelist = []
	for vc in c_list:
		l = [] 
		for vx in x_list:
			if TYPE == "t":
				i = current(vx, vc) # Vg, Vd
			if TYPE == "f":
				i = current(vc, vx) # Vg, Vd
			if LOG:
				l.append(abs(i))
			else:
				l.append(i)
		if LOG:
			plt.semilogy(x_list, l, 'b-')
		else:
			plt.plot(x_list, l, 'b-')
	savelist.append(l)
	plt.show()
	return savelist

def saveData(label,fData, c_list, r_list, save_list, scale, c_list_label):
	with open(label + "_" + fData + '_trained.csv', 'wb') as csvfile:
		iwriter = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
		firstRow = [c_list_label]
		firstRow.extend(c_list)
		iwriter.writerow(firstRow)
		for i in range(len(r_list)): 
			row = []
			row.append(r_list[i])
			for j in range(len(c_list)):
				row.append(save_list[j][i] * scale)
			iwriter.writerow(row)



if __name__ == "__main__":
	pass