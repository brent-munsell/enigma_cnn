from functions import optimal_grid_parameters, grid_results
import pickle
import numpy as np

folder = './run_grid_opts'
D, opts = grid_results(folder)

a = D['A'].mean(axis=1)
b = D['A'].mean(axis=1) - D['A'].std(axis = 1)
opt_params_stats = {'max_mean_acc' : a.max(), 'max_ma_D_idx' : np.argmax(a), 'max_mean_acc_min_std' : b.max(), 'max_mams_D_idx' : np.argmax(b)}

f = open('./optimal_params/opt_params_opts.pkl',"wb")
pickle.dump(opts, f)
f.close()

f = open('./optimal_params/opt_params_opts.txt',"w")
f.write(str(opts))
f.close()

f = open('./optimal_params/opt_params_D.pkl',"wb")
pickle.dump(D, f)
f.close()

f = open('./optimal_params/opt_params_D.txt',"w")
f.write(str(D))
f.close()

f = open('./optimal_params/opt_params_stats.pkl',"wb")
pickle.dump(opt_params_stats, f)
f.close()

f = open('./optimal_params/opt_params_stats.txt',"w")
f.write(str(opt_params_stats))
f.close()