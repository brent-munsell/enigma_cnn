from functions import process_results
import pickle
import numpy as np

opts_dir = './run_opt_cv_opts'
nets_dir = './run_opt_cv_nets'
C, A = process_results(opts_dir, nets_dir)

metrics_stats = {'ac_mean' : C['ac'].mean(axis=0), 'ac_std' : C['ac'].std(axis=0), 'ac_mean_minus_std' : C['ac'].mean(axis=0) - C['ac'].std(axis=0), 'ppv_mean' : C['ppv'].mean(axis=0), 'ppv_std' : C['ppv'].std(axis=0), 'ppv_mean_minus_std' : C['ppv'].mean(axis=0) - C['ppv'].std(axis=0), 'npv_mean' : C['npv'].mean(axis=0), 'npv_std' : C['npv'].std(axis=0), 'npv_mean_minus_std' : C['npv'].mean(axis=0) - C['npv'].std(axis=0), 'sen_mean' : C['sen'].mean(axis=0), 'sen_std' : C['sen'].std(axis=0), 'sen_mean_minus_std' : C['sen'].mean(axis=0) - C['sen'].std(axis=0), 'spc_mean' : C['spc'].mean(axis=0), 'spc_std' : C['spc'].std(axis=0), 'spc_mean_minus_std' : C['spc'].mean(axis=0) - C['spc'].std(axis=0), 'auc_mean' : C['auc'].mean(axis=0), 'auc_std' : C['auc'].std(axis=0), 'auc_mean_minus_std' : C['auc'].mean(axis=0) - C['auc'].std(axis=0), 'ax_mean' : C['ax'].mean(axis=0), 'ax_std' : C['ax'].std(axis=0), 'ax_mean_minus_std' : C['ax'].mean(axis=0) - C['ax'].std(axis=0), 'ay_mean' : C['ay'].mean(axis=0), 'ay_std' : C['ay'].std(axis=0), 'ay_mean_minus_std' : C['ay'].mean(axis=0) - C['ay'].std(axis=0)}

f = open('./opt_cv_results/opt_cv_C.pkl',"wb")
pickle.dump(C, f)
f.close() 

f = open('./opt_cv_results/opt_cv_C.txt',"w")
f.write(str(C))
f.close()

f = open('./opt_cv_results/opt_cv_A.pkl',"wb")
pickle.dump(A, f)
f.close()

f = open('./opt_cv_results/opt_cv_A.txt',"w")
f.write(str(A))
f.close()

f = open('./opt_cv_results/opt_cv_metric_stats.pkl',"wb")
pickle.dump(metric_stats, f)
f.close()

f = open('./opt_cv_results/opt_cv_metric_stats.txt',"w")
f.write(str(metric_stats))
f.close()