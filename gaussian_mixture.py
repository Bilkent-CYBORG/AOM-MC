import numpy as np
import itertools

import pickle

from matplotlib.ticker import MaxNLocator
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn import mixture
import seaborn as sns

# get location data
from tqdm import tqdm

import fs_loader
sns.set(style='white')


with open(fs_loader.saved_df_filename, 'rb') as input_file:
    df = pickle.load(input_file)
X = df[['lat', 'long']].values

print("Fitting Gaussian Mixtures using EM on data from {}...".format(fs_loader.saved_df_filename))

lowest_bic = np.infty
bic = []
n_components_range = range(1, 31)
cv_types = ['Full']
for cv_type in cv_types:
    for n_components in tqdm(n_components_range):
        # Fit a Gaussian mixture with EM
        gmm = mixture.GaussianMixture(n_components=n_components,
                                      covariance_type=cv_type.lower())
        gmm.fit(X)
        bic.append(gmm.bic(X))
        if bic[-1] < lowest_bic:
            lowest_bic = bic[-1]
            best_gmm = gmm

bic = np.array(bic)
color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue',
                              'darkorange'])
clf = best_gmm
with open('gmm_{}'.format(fs_loader.city_name), 'wb') as output:
    pickle.dump(best_gmm, output, pickle.HIGHEST_PROTOCOL)
bars = []
print('Lowest BIC corresponds to {} components'.format(best_gmm.n_components))

# Plot the BIC scores
fig = plt.figure()
plt.bar(n_components_range, bic, align='center', alpha=0.5)
# for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
#     xpos = np.array(n_components_range) + .2 * (i - 2)
#     bars.append(plt.bar(xpos, bic[i * len(n_components_range):
#                                   (i + 1) * len(n_components_range)],
#                         width=.2))
# plt.xticks(n_components_range)
# plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
# #plt.title('BIC score per model')
# xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 +\
#     .2 * np.floor(bic.argmin() / len(n_components_range))
# plt.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
# plt.legend([b[0] for b in bars], cv_types)

plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.xlabel('Number of components')
plt.xlabel('Bayesian information criterion')
ax = fig.gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.tight_layout()
plt.savefig("bic_scores_{}.pdf".format(fs_loader.city_name), bbox_inches='tight', pad_inches=0.01)


# Plot the winner
colors = sns.color_palette("hls", clf.n_components)
_, ax = plt.subplots(figsize=(6, 6))
ax.set_aspect('equal')
Y_ = clf.predict(X)
for i, (mean, cov, color) in enumerate(zip(clf.means_, clf.covariances_,
                                           colors)):
    v, w = linalg.eigh(cov)
    if not np.any(Y_ == i):
        continue
    result = ax.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)
    # color = result.get_facecolor()[0]

    # Plot an ellipse to show the Gaussian component
    angle = np.arctan2(w[0][1], w[0][0])
    angle = 180. * angle / np.pi  # convert to degrees
    v = 2. * np.sqrt(2.) * np.sqrt(v)
    ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
    ell.set_clip_box(ax.bbox)
    ell.set_alpha(.5)
    ax.add_artist(ell)

# plt.xticks(())
# plt.yticks(())
# plt.subplots_adjust(hspace=.35, bottom=.02)
plt.xlabel('Scaled latitude')
plt.ylabel('Scaled longitude')
plt.tight_layout()
plt.savefig("gm_{}.pdf".format(fs_loader.city_name), bbox_inches='tight', pad_inches=0.01)
plt.show()