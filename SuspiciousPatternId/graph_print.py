from Definitions import *

percentile_under_tol = np.load(str(tol) + '/' + str(tol) + 'percentile_under_tol.npy', allow_pickle=True)

plt.hist(percentile_under_tol, bins=100)
plt.yscale('log')
plt.savefig(str(tol) + '/' + str(trimming) + '-percentile_under_tol.png')
plt.show()