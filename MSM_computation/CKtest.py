import numpy as np
import pyemma
import matplotlib.pyplot as plt
import pickle
import mplhelv 

#Load the MSM object
msmAPO = pickle.load(open('./msmobj/msmobj_100_0.95_5.pkl','rb'))

#Perform CKTest for Apo-SMO and estimate errors as well
ck_APO = msmAPO.cktest(5,err_est=True) 

#Plot the cktest
cktest_plot = pyemma.plots.plot_cktest(ckAPO)
cktest_plot[0].savefig('./CKTEST_APO_with_errors.png', transparent=True,dpi=300)