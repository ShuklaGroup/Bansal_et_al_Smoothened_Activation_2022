### SCRIPT TO MAKE MSM OBJECT FROM THE CLUSTERED DATA AND COMPUTE IMPLIED TIMESCALES
###  Written by Prateek Bansal
### pdb3@illinois.edu
### October 27, 2021

import numpy as np
import pyemma
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import seaborn as sns
clus = [100,200,400,600,800,1000,1200,1400,1600,1800]
cutoffs = [0.95,0.8,0.65]
for clu in tqdm(clus):
  for ct in tqdm(cutoffs):
    cluster_tica_dtrajs = pickle.load(open(f'clus_tica_dtrajs_{clu}_{ct}.pkl','rb'))
    msm=pyemma.msm.estimate_markov_model(dtrajs=cluster_tica_dtrajs,lag=300,score_method='VAMP2',score_k=5)
    pickle.dump(msm,open(f'msmobj_{clu}_{ct}_5.pkl','wb'))
    #msm.save(f'msmobj_{clu}_{ct}.pkl',overwrite=True)
    score=msm.score_cv(dtrajs=cluster_tica_dtrajs)
    np.save(f'VAMP2_{clu}_{ct}_5.npy',score)
