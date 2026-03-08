import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np
import arviz as az

# let's try to do an approximate FOM ...
def DE_fom(test_chain,burnin):

    test_chain = test_chain[:,burnin:,:].reshape((-1,test_chain.shape[2]))
    w0wa_samps = test_chain[:,[2,3]]

    # choose a random 10000 samps to fit from (there's way too many samples to do the fitting)
    if np.shape(w0wa_samps)[0] > 100000:
        idx_avail = np.arange(0,len(w0wa_samps))
        idx_chosen = np.random.choice(idx_avail,100000,replace=False)
        w0wa_samps_small = w0wa_samps[idx_chosen]
    else:
        w0wa_samps_small = w0wa_samps

    # fit gaussian to the samps
    Mean = np.mean(w0wa_samps_small,axis=0)
    Cov = np.cov(w0wa_samps_small,rowvar=False)

    # find the pivot value
    one_minus_ap = - Cov[1,0] / Cov[1,1] # 1 - ap = - <delta_w0 delta_wa> / <delta_wa^2>
    z_pivot = one_minus_ap / (-one_minus_ap+1)
    print('pivot redshift: ', one_minus_ap / (-one_minus_ap+1))
    # transform Mu and Cov
    Transform = np.asarray([[1, one_minus_ap],[0, 1]])
    Mean_pivot = np.matmul(Transform,Mean)
    Cov_pivot = np.matmul(Transform,np.matmul(Cov,Transform.T))
    print('DE FOM: ', 1/(np.sqrt(Cov_pivot[0,0]*Cov_pivot[1,1])))
    de_fom = 1/(np.sqrt(Cov_pivot[0,0]*Cov_pivot[1,1]))

    return z_pivot, de_fom


def twoD_area(emcee_chain,param_idxs,burnin,num_gridpoints=100):
    # written with help from chatGPT

    # Extract 2D samples
    num_params = emcee_chain.shape[2]
    chain = emcee_chain[:,burnin:,:].reshape((-1,num_params))
    x = chain[:, param_idxs[0]]
    y = chain[:, param_idxs[1]]

    # Step 2: KDE estimation over a grid
    kde = gaussian_kde(np.vstack([x, y]))

    # Create a grid
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    X, Y = np.meshgrid(np.linspace(xmin,xmax,num_gridpoints),
        np.linspace(ymin,ymax,num_gridpoints))
    positions = np.vstack([X.ravel(), Y.ravel()])
    Z = kde(positions).reshape(X.shape)

    # Step 3: Find the 68% highest posterior density (HPD) threshold
    Z_flat = Z.flatten()
    Z_sorted = np.sort(Z_flat)[::-1]
    cumulative = np.cumsum(Z_sorted)
    cumulative /= cumulative[-1]

    # Find the density value at 68% cumulative probability
    threshold_index = np.searchsorted(cumulative, 0.68)
    density_threshold = Z_sorted[threshold_index]

    # Step 4: Binary mask of the 68% region
    mask = Z >= density_threshold

    # Step 5: Estimate area using pixel counting
    dx = (xmax - xmin) / num_gridpoints
    dy = (ymax - ymin) / num_gridpoints
    pixel_area = dx * dy
    area_68 = mask.sum() * pixel_area

    return area_68


def median_and_uncertainty(emcee_chain,burnin):

    num_params = emcee_chain.shape[2]
    chain = emcee_chain[:,burnin:,:].reshape((-1,num_params))

    med = np.median(chain,axis=0)
    low = np.quantile(chain,q=0.1586,axis=0)
    high = np.quantile(chain,q=0.8413,axis=0)

    for i,post_med in enumerate(med):
        print(np.around(post_med,3), ' + ', np.around(high[i]-post_med,3), ' - ', np.around(post_med-low[i],3))
        arviz_hdi = az.hdi(chain[:,i], hdi_prob=.68)
        #print('arviz hdi: ', arviz_hdi[1] - arviz_hdi[0])
        print('arviz (68 hdi / 2)', 
            np.round((arviz_hdi[1] - arviz_hdi[0])/2, 3))
        if i == 1:
            break


def analyze_chains(emcee_chain,param_labels,true_hyperparameters,
                    outfile,show_chains=False,
                    burnin=int(1e3)):
    """
    Args:
        emcee_chain (array[n_walkers,n_samples,n_params])
    """

    if show_chains:
        for i in range(emcee_chain.shape[2]):
            plt.figure()
            #indices = np.arange(0,emcee_chain.shape[1],10)
            plt.plot(emcee_chain[:,:,i].T,'.')
            plt.title(param_labels[i])
            plt.show()

    num_params = emcee_chain.shape[2]
    chain = emcee_chain[:,burnin:,:].reshape((-1,num_params))

    labels = ['Ground Truth', 'Inferred Value', 'Bias in $\sigma$', 'Fractional Error']

    med = np.median(chain,axis=0)
    low = np.quantile(chain,q=0.1586,axis=0)
    high = np.quantile(chain,q=0.8413,axis=0)

    error = med - true_hyperparameters
    sigma = ((high-med)+(med-low))/2
    bias = error/sigma

    metrics = [true_hyperparameters,med,bias,error]

    with open(outfile,'w') as f:

        f.write('\hline')
        f.write('\n')

        for i,lab in enumerate(labels):
            f.write(lab)
            f.write(' ')

            for k,m in enumerate(metrics[i]):
                f.write('& ')
                f.write(str(np.around(m,2)))
                f.write(' ')
                if i == 1:
                    f.write('$\pm$' + str(np.around(sigma[k],2)))

            f.write(r'\\')
            f.write('\n')
            f.write('\hline')
            f.write('\n')


    for j in range(num_params):

        med = np.median(chain[:,j])
        low = np.quantile(chain[:,j],q=0.1586)
        high = np.quantile(chain[:,j],q=0.8413)
        print(param_labels[j])
        print("\t", round(med,3), "+", round(high-med,3), "-", round(med-low,3))
        error = med - true_hyperparameters[j]
        if error > 0:
            bias = error/round(med-low,3)
        else:
            bias = error/round(high-med,3)
        print("\t", "Bias in Std. Devs: ", round(bias,3))

        frac_error = error/true_hyperparameters[j]
        print("\t","Fractional Error: ",round(frac_error,3))