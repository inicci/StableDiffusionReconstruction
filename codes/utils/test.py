import numpy as np
IMGIDX=0
s_z = np.load(r'..\..\decoded\subj01\subj01_early_scores_init_latent.npy')[IMGIDX]  # (6400,)
s_c = np.load(r'..\..\decoded\subj01\subj01_ventral_scores_c.npy')[IMGIDX]         # (59136,)
print('z0 mean/std:', s_z.mean(), s_z.std())
print('c  mean/std:', s_c.mean(), s_c.std())
