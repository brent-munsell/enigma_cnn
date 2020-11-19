from functions import optimal_slice, slice_results
import pickle
import numpy as np

folder = './run_slice_opts'
D, opts = slice_results(folder)

a = D['A'].mean(axis=1)
b = D['A'].mean(axis=1) - D['A'].std(axis = 1)
opt_slice_stats = {'max_mean_acc' : a.max(), 'max_ma_D_idx' : np.argmax(a), 'max_mean_acc_min_std' : b.max(), 'max_mams_D_idx' : np.argmax(b)}

f = open('./optimal_slice/opt_slice_opts.pkl',"wb")
pickle.dump(opts, f)
f.close() 

f = open('./optimal_slice/opt_slice_opts.txt',"w")
f.write(str(opts))
f.close()

f = open('./optimal_slice/opt_slice_D.pkl',"wb")
pickle.dump(D, f)
f.close() 

f = open('./optimal_slice/opt_slice_D.txt',"w")
f.write(str(D))
f.close()

f = open('./optimal_slice/opt_slice_stats.pkl',"wb")
pickle.dump(opt_slice_stats, f)
f.close()

f = open('./optimal_slice/opt_slice_stats.txt',"w")
f.write(str(opt_slice_stats))
f.close()