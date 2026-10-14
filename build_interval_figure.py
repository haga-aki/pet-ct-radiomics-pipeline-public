"""Paired PET profiles and held-out boundary measurements for all five cases."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.ndimage import gaussian_filter1d


def read(path):
    with open(path) as stream:
        return list(csv.DictReader(stream))


def build(root):
    cases = [root/f'Case_{i:03d}' for i in range(1, 6)]
    for case in cases:
        if not (case/'manifest.json').is_file():
            raise ValueError(f'Incomplete case: {case}')
    figure, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True,
                                gridspec_kw={'width_ratios': [1.2, 1, 1]})
    colors = ['#0072B2', '#009E73', '#D55E00', '#CC79A7', '#56B4E9']
    summaries = []
    for row, organ in enumerate(('liver', 'spleen')):
        for i, case in enumerate(cases):
            patches = [p for p in read(case/'paired_patches.csv') if p['organ']==organ and p['split']=='held_out']
            profiles = [p for p in read(case/'paired_profiles.csv') if p['organ']==organ and p['split']=='held_out']
            x = sorted(set(float(p['distance_mm']) for p in profiles))
            for variant, style in [('before', '-'), ('after', '--')]:
                curve = np.array([np.nanmedian([float(p[variant]) for p in profiles if float(p['distance_mm']) == d]) for d in x])
                finite = np.isfinite(curve)
                # Do not manufacture extrapolated plotting data.
                axes[row, 0].plot(np.asarray(x)[finite], gaussian_filter1d(curve[finite], .7, mode='nearest'),
                                  style, color=colors[i], lw=1.5)
            summary = dict(case=case.name, organ=organ, heldout_patches=len(patches))
            for column, stem in [(1, 'interval_error_mm'), (2, 'mm')]:
                arrays = [np.array([float(p[f'{v}_{stem}']) for p in patches]) for v in ('before', 'after')]
                paired = np.isfinite(arrays[0]) & np.isfinite(arrays[1])
                summary[f'{stem}_finite_pairs'] = int(paired.sum())
                if not paired.any():
                    raise ValueError(f'{case.name}/{organ}: no finite paired values')
                b, a = [v[paired] for v in arrays]
                summary[f'{stem}_before_median_absolute'] = float(np.median(abs(b)))
                summary[f'{stem}_after_median_absolute'] = float(np.median(abs(a)))
                summary[f'{stem}_improved'] = int((abs(a)<abs(b)-1e-8).sum())
                summary[f'{stem}_worsened'] = int((abs(a)>abs(b)+1e-8).sum())
                summary[f'{stem}_unchanged'] = int((abs(abs(a)-abs(b))<=1e-8).sum())
                for offset, array, marker, variant in zip((-.16, .16), (b, a), ('o', 's'), ('before', 'after')):
                    q1, median, q3 = np.percentile(array, [25, 50, 75])
                    summary[f'{stem}_{variant}_median'] = float(median)
                    summary[f'{stem}_{variant}_q25'] = float(q1)
                    summary[f'{stem}_{variant}_q75'] = float(q3)
                    axes[row, column].errorbar(median, i+offset, xerr=[[median-q1], [q3-median]], fmt=marker,
                                              color=colors[i], markerfacecolor='white' if variant=='before' else colors[i],
                                              capsize=3, markersize=6)
            summaries.append(summary)
        axes[row, 0].axvline(0, color='gray', lw=.8)
        axes[row, 0].axhline(.5, color='gray', lw=.8, linestyle=':')
        axes[row, 0].set(xlim=(-20, 20), ylim=(-.15, 1.25),
                         xlabel='Distance from organ mask surface (mm)', ylabel='Normalized PET uptake')
        for col in (1, 2):
            axes[row, col].set_yticks(range(5), [f'Case {i}' for i in range(1, 6)])
            axes[row, col].invert_yaxis()
            axes[row, col].axvline(0, color='gray', lw=.8)
            axes[row, col].grid(axis='x', alpha=.2)
        axes[row, 1].set_xlabel('Signed distance outside interface interval (mm)')
        axes[row, 2].set_xlabel('PET 50% − organ mask surface (mm)')
        for col, text in enumerate(('PET transition', 'Interface correspondence', 'Organ surface offset')):
            axes[row, col].set_title(f'{chr(65+row*3+col)}  {organ.capitalize()}: {text}', fontsize=11, loc='left')
    legends = [Line2D([0], [0], color=color, label=f'Case {i+1}') for i, color in enumerate(colors)]
    legends += [Line2D([0], [0], color='black', ls='-', label='Original'),
                Line2D([0], [0], color='black', ls='--', label='Adapted')]
    figure.legend(handles=legends, loc='outside lower center', ncols=7, frameon=False)
    figure.savefig(root/'paired_interface_profiles.png', dpi=300)
    figure.savefig(root/'paired_interface_profiles.tiff', dpi=300, pil_kwargs={'compression': 'tiff_lzw'})
    plt.close(figure)
    with open(root/'heldout_interval_summary.csv', 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    build(parser.parse_args().root)
