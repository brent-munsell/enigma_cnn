from functions import process_slice
import pickle
import numpy as np

opts_dir = './run_slice_opts'
nets_dir = './run_slice_nets'
C, pslice = process_slice(opts_dir, nets_dir)

metric_stats = []
for i in C:
    metric_stats.append({'ac_mean' : i['ac'].mean(axis=0), 'ac_std' : i['ac'].std(axis=0), 'ac_mean_minus_std' : i['ac'].mean(axis=0) - i['ac'].std(axis=0), 'ppv_mean' : i['ppv'].mean(axis=0), 'ppv_std' : i['ppv'].std(axis=0), 'ppv_mean_minus_std' : i['ppv'].mean(axis=0) - i['ppv'].std(axis=0), 'npv_mean' : i['npv'].mean(axis=0), 'npv_std' : i['npv'].std(axis=0), 'npv_mean_minus_std' : i['npv'].mean(axis=0) - i['npv'].std(axis=0)})

f = open('./slice_results/slice_C.pkl',"wb")
pickle.dump(C, f)
f.close()

f = open('./slice_results/slice_C.txt',"w")
f.write(str(C))
f.close()

f = open('./slice_results/slice_pslice.pkl',"wb")
pickle.dump(pslice, f)
f.close()

f = open('./slice_results/slice_pslice.txt',"w")
f.write(str(pslice))
f.close()

f = open('./slice_results/slice_metric_stats.pkl',"wb")
pickle.dump(metric_stats, f)
f.close()

f = open('./slice_results/slice_metric_stats.txt',"w")
f.write(str(metric_stats))
f.close()

