"""Standalone scientific diagnostic; unit-wrench capacity is not measured force."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/track3/episode_000011/contact_wrench_probe'
z=np.load(OUT/'provisional_reference.npz');d=z['nearest_link_distance_m'];support=z['support'];frames=z['frame_index']
fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained');colors=['#355fc5','#dd8b21','#168a77']
for threshold,color in zip(z['thresholds_m'],colors):
    axes[0,0].plot(frames,np.sum(d<threshold,axis=(1,2,3)),label=f'{threshold*1000:g} mm',color=color,lw=1.5)
axes[0,0].set(title='Geometric proximity depends on threshold',xlabel='Frame',ylabel='Active MANO-link / object pairs');axes[0,0].legend(ncol=3)
for mi,mu in enumerate(z['friction_coefficients']):
    values=support[1,mi];coverage=np.mean(values>1e-6,axis=-1);coverage[~z['observed_hand']]=np.nan
    axes[0,1].plot(frames,np.nanmean(coverage,axis=(1,2)),label=f'mu = {mu:g}',color=colors[mi],lw=1.5)
axes[0,1].set(title='Positive support directions (10 mm candidate)',xlabel='Frame',ylabel='Fraction of shared 512 directions',ylim=(0,1));axes[0,1].legend()
for hi,side in enumerate(['Left','Right']):
    values=support[1,1,:,hi];magnitude=np.nanmean(values,axis=(1,2),where=np.isfinite(values));axes[1,0].plot(frames,magnitude,label=side,lw=1.5)
axes[1,0].set(title='Unit-wrench support (10 mm, mu = 0.3)',xlabel='Frame',ylabel='Mean support; dimensionless');axes[1,0].legend()
axes[1,1].imshow(z['observed_hand'].T,aspect='auto',interpolation='nearest',cmap='Blues',vmin=0,vmax=1,extent=(-.5,136.5,1.5,-.5));axes[1,1].set(yticks=[0,1],yticklabels=['Left','Right'],xlabel='Frame',title='Observed hand availability; missing remains unknown')
for ax in axes.flat:ax.spines[['top','right']].set_visible(False)
fig.suptitle('Provisional CHORD contact/wrench diagnostic — geometry and hand uncertainty unresolved',fontsize=12)
fig.savefig(OUT/'contact_wrench_uncertainty.png',dpi=150);fig.savefig(OUT/'contact_wrench_uncertainty.pdf');plt.close(fig)
print('Saved contact/wrench PNG and PDF')
