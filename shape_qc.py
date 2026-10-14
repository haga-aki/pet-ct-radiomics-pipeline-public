"""Descriptive contour-change checks on saved masks, without smoothing masks."""
import argparse
import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np


def top_face(mask, spacing):
    occupied = mask.any(axis=2)
    height = (mask.shape[2] - .5 - np.argmax(mask[:, :, ::-1], axis=2))*spacing
    return np.where(occupied, height, np.nan)


def audit_shape(case):
    manifest = json.loads((case/'manifest.json').read_text())
    field = np.load(case/'displacement_field.npz')
    rows = []
    for organ in ('liver', 'spleen'):
        source = nib.load(manifest['inputs']['masks'][organ]['path'])
        original = top_face(np.asarray(source.dataobj)>0, source.header.get_zooms()[2])
        adapted = top_face(np.asarray(nib.load(str(case/'masks'/f'{organ}.nii.gz')).dataobj)>0,
                           source.header.get_zooms()[2])
        # Use original anatomy and the recorded support, not the largest effect.
        valid = np.isfinite(original) & np.isfinite(adapted) & (np.abs(original-field['center_mm']) <= 6.) & (field['shift_mm'] != 0)
        delta = adapted-original
        slopes, old_bend, new_bend = [], [], []
        for axis in (0, 1):
            step = max(1, round(10/source.header.get_zooms()[axis]))
            left, middle, right = [], [], []
            for a in (0, 1):
                left.append(slice(None, -2*step) if a==axis else slice(None))
                middle.append(slice(step, -step) if a==axis else slice(None))
                right.append(slice(2*step, None) if a==axis else slice(None))
            l, m, r = tuple(left), tuple(middle), tuple(right)
            use = valid[l] & valid[m] & valid[r]
            physical = step*source.header.get_zooms()[axis]
            slopes.extend(((delta[r]-delta[l])/(2*physical))[use])
            old_bend.extend(((original[r]-2*original[m]+original[l])/physical**2)[use])
            new_bend.extend(((adapted[r]-2*adapted[m]+adapted[l])/physical**2)[use])
        rows.append(dict(organ=organ, supported_columns=int(valid.sum()), triples=len(old_bend),
                         displacement_slope_rms=float(np.sqrt(np.mean(np.square(slopes)))) if slopes else None,
                         original_height_second_difference_rms_per_mm=float(np.sqrt(np.mean(np.square(old_bend)))) if old_bend else None,
                         adapted_height_second_difference_rms_per_mm=float(np.sqrt(np.mean(np.square(new_bend)))) if new_bend else None))
    result = dict(case=case.name, organs=rows,
                  interpretation='10-mm finite differences of SI height; descriptive, not a curvature or anatomical-accuracy validation')
    (case/'shape_qc.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case', type=Path)
    audit_shape(parser.parse_args().case)
