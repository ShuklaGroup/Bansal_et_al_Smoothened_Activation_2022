### A SCRIPT TO MAKE MORE SCRIPTS, FOR DIFFERENT CLUSTER NUMBERS (k) and DIFFERENT VARIATIONAL CUTOFFS, TO PERFORM CLUSTERING IN PARALLEL
### WRITTEN BY PRATEEK BANSAL
### OCT 15, 2021


import numpy as np
import pyemma
import matplotlib.pyplot as plt
import pickle
from tqdm.notebook import tqdm
t = np.load('totdist_RRCSg40.npy',allow_pickle=True)  ## The actual distances file, which consists of all the features calculated for constructing the MSM (n = 58). The shape of the numpy array is (number of trajs, no. of frames, no. of features) 
lags=np.append(np.arange(10,300,10),np.arange(300,600,10))
impl=[]
t = t.tolist()
clus = [100,200,400,600,800,1000,1200,1400,1600,1800]
cutoffs = [0.95,0.8,0.65]
def do_tica_given_lagtime():
    tica_lagtime = 300
    for ct in tqdm(cutoffs):
        tica = pyemma.coordinates.tica(t,lag=tica_lagtime,var_cutoff=ct)
        tica_output = tica.get_output()
        pickle.dump(tica,open(f'ticaobj_{ct}.pkl','wb'))
        pickle.dump(tica_output,open(f'ticaoutput_{ct}.pkl','wb'))
def master_scripts_gen():
    g=open('clustering_script_list','w+')
    for clu in tqdm(clus):
        for ct in tqdm(cutoffs):
            f=open(f'clustering_{clu}_{ct}.py','w+')
            g.write(f'clustering_{clu}_{ct}.py\n')
            f.write(f'''
import numpy as np
import pyemma
import matplotlib.pyplot as plt
import pickle
from tqdm.notebook import tqdm
lags=np.append(np.arange(10,300,10),np.arange(300,600,10))
impl=[]
clu = {clu}
ct = {ct}
tica_output = pickle.load(open(f'ticaoutput_{{ct}}.pkl','rb'))
cluster_tica = pyemma.coordinates.cluster_kmeans(tica_output, k=clu, max_iter=100, stride=1)
pickle.dump(cluster_tica,open(f'clusobj_{{clu}}_{{ct}}.pkl','wb'))
cluster_tica.save(f'clusobj_{{clu}}_{{ct}}_pyemma.pkl')
cluster_tica_output = cluster_tica.get_output()
pickle.dump(cluster_tica_output, open(f'cluster_tica_output_{{clu}}_{{ct}}.pkl','wb'))
cluster_tica_dtrajs = cluster_tica.dtrajs
pickle.dump(cluster_tica_dtrajs,open(f'clus_tica_dtrajs_{{clu}}_{{ct}}.pkl','wb'))
its = pyemma.msm.its(cluster_tica_dtrajs,lags=lags)
its.save(f'its_{{clu}}_{{ct}}.pkl')''')
            f.close()
    g.close()
if __name__=='__main__':
    do_tica_given_lagtime()
    master_scripts_gen() 
