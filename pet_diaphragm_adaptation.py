"""Experimental PET-guided, local SI adaptation of CT-derived diaphragm masks.

PET is never deformed. Original masks are never overwritten. A shared, monotone
inverse warp moves liver, spleen and lung labels together. This is PET-boundary
adaptation, not correction of attenuation artefacts or validated anatomy.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter, gaussian_filter1d
from scipy.spatial import ConvexHull, Delaunay, QhullError

LABELS = ("liver", "spleen", "lung_lower_lobe_left", "lung_lower_lobe_right",
          "lung_upper_lobe_left", "lung_upper_lobe_right", "lung_middle_lobe_right")


@dataclass(frozen=True)
class Parameters:
    patch_mm: float = 10.0
    min_columns: int = 20
    profile_sigma_voxels: float = 0.7
    spatial_sigma_mm: float = 10.0
    max_shift_mm: float = 9.0
    band_mm: float = 36.0
    support_distance_mm: float = 15.0
    minimum_inside_suv: float = 1.0
    minimum_contrast_suv: float = 0.35
    training_parity: int = 0
    adaptation_mode: str = "interface_interval"
    maximum_interface_gap_mm: float | None = 6.0
    deadband_mm: float = 1.5
    organ_bias_gate_mm: float | None = 1.5
    minimum_training_patches: int = 5
    field_model: str = "shape_regularized"
    control_spacing_mm: float = 20.0
    residual_sigma_mm: float = 20.0
    residual_penalty: float = 1.0
    maximum_local_residual_mm: float = 3.0
    huber_delta_mm: float = 1.5
    minimum_unambiguous_fraction: float = 0.75
    maximum_crossing_iqr_mm: float = 6.0
    require_consistent_direction: bool = True


def interval_error(position, lower, upper):
    """Signed distance to a closed anatomical interface interval, zero within."""
    if np.any(np.asarray(upper) < np.asarray(lower)):
        raise ValueError("Reversed interface boundaries")
    distance = np.where(position < lower, position-lower,
                        np.where(position > upper, position-upper, 0.))
    return np.where(np.isfinite(position) & np.isfinite(lower) & np.isfinite(upper), distance, np.nan)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def crossings(profile, spacing, sigma):
    """Nearest descending crossings relative to the original CT voxel face."""
    smooth = gaussian_filter1d(np.asarray(profile, float), sigma, mode="nearest")
    positions = (np.arange(-9, 10) - 0.5) * spacing
    result = {}
    for threshold in (0.1, 0.2, 0.5, 0.8, 0.9):
        candidates = []
        for j in range(len(smooth) - 1):
            if smooth[j] >= threshold > smooth[j + 1]:
                fraction = (threshold - smooth[j]) / (smooth[j + 1] - smooth[j])
                candidates.append(positions[j] + spacing * fraction)
        result[threshold] = min(candidates, key=abs) if candidates else np.nan
    return result


def get_patches(pet, organ, lung, spacing, parameters):
    xy = np.argwhere(organ.any(axis=2))
    groups = {}
    gaps = {}
    # Absolute image-index patch origin, fixed before and after correction.
    for x, y in xy:
        occupied = np.flatnonzero(organ[x, y])
        if len(occupied) < 7:
            continue
        k = occupied[-1]
        if k < 10 or k + 10 >= organ.shape[2]:
            continue
        if lung[x, y, k + 1:k + 8].sum() < 2:
            continue
        floor = k + 1 + np.flatnonzero(lung[x, y, k+1:k+8])[0]
        gap = (floor-k-1)*spacing[2]
        if parameters.maximum_interface_gap_mm is not None and gap > parameters.maximum_interface_gap_mm:
            continue
        raw = pet[x, y, k + np.arange(-9, 10)].astype(float)
        inside, outside = np.median(raw[2:7]), np.median(raw[11:17])
        if (not np.isfinite(raw).all() or inside < parameters.minimum_inside_suv
                or inside - outside < parameters.minimum_contrast_suv):
            continue
        patch = tuple(np.floor(np.array([x, y]) * spacing[:2] / parameters.patch_mm).astype(int))
        groups.setdefault(patch, []).append((x, y, int(k), (raw - outside) / (inside - outside)))
        gaps.setdefault(patch, []).append(float(gap))
    patches = []
    for key, columns in sorted(groups.items()):
        if len(columns) < parameters.min_columns:
            continue
        profiles = np.array([row[3] for row in columns])
        c = crossings(np.median(profiles, axis=0), spacing[2], parameters.profile_sigma_voxels)
        if not all(np.isfinite(list(c.values()))):
            continue
        if not (c[0.8] <= c[0.5] <= c[0.2] and c[0.2] > c[0.8] and c[0.1] > c[0.9]):
            continue
        gap = float(np.median(gaps[key]))
        estimate = (float(interval_error(c[.5], 0., gap))
                    if parameters.adaptation_mode == "interface_interval" else c[.5])
        if abs(estimate) <= parameters.deadband_mm:
            estimate = 0.
        reliability = profile_reliability(profiles, spacing[2], parameters.profile_sigma_voxels)
        patches.append(dict(key=key, columns=columns, shift=c[0.5], correction=estimate,
                            gap_mm=gap, width=c[0.2]-c[0.8],
                            **reliability,
                            training=sum(key) % 2 == parameters.training_parity))
    return patches


def profile_reliability(profiles, spacing, sigma):
    """Fitting weight, not a statistical confidence interval or clinical cutoff."""
    smooth = gaussian_filter1d(np.asarray(profiles, float), sigma, axis=1, mode="nearest")
    descending = (smooth[:, :-1] >= .5) & (smooth[:, 1:] < .5)
    unique = descending.sum(axis=1) == 1
    positions = []
    for row in np.flatnonzero(unique):
        j = np.flatnonzero(descending[row])[0]
        fraction = (.5 - smooth[row, j]) / (smooth[row, j+1] - smooth[row, j])
        positions.append((j - 9 - .5 + fraction) * spacing)
    fraction = float(np.mean(unique))
    iqr = float(np.diff(np.percentile(positions, [25, 75]))[0]) if positions else float("inf")
    return dict(unambiguous_fraction=fraction, crossing_iqr_mm=iqr,
                reliability=fraction / (1 + (iqr / spacing)**2))


def weighted_quantile(values, weights, quantile):
    order = np.argsort(values, kind="stable")
    values, weights = np.asarray(values)[order], np.asarray(weights)[order]
    return float(values[np.searchsorted(np.cumsum(weights), weights.sum() * quantile)])


def weighted_median(values, weights):
    return weighted_quantile(values, weights, .5)


def radial_basis(points, controls, sigma):
    distance2 = ((points[:, None, :] - controls[None, :, :])**2).sum(axis=2)
    basis = np.exp(-distance2 / (2 * sigma**2))
    return basis / np.maximum(basis.sum(axis=1, keepdims=True), 1e-15)


def envelope_support(weights, spacing, distance_mm):
    """Taper outside the fitting envelope, never inside held-out patch gaps.

    The convex envelope affects field support only. It does not fill mask holes
    or use held-out PET values. Degenerate point/line support falls back to its
    sampled support, which is relevant to small synthetic unit-test fixtures.
    """
    core = weights > 0
    points = np.argwhere(core)
    if len(points) >= 3:
        try:
            hull = ConvexHull(points)
            grid = np.indices(weights.shape).reshape(2, -1).T
            core = (Delaunay(points[hull.vertices]).find_simplex(grid) >= 0).reshape(weights.shape)
        except QhullError:
            pass
    distance = distance_transform_edt(~core, sampling=spacing[:2])
    fraction = np.clip(distance/distance_mm, 0, 1)
    return .5*(1+np.cos(np.pi*fraction))


def fit_field_shape(patches, shape, spacing, parameters):
    """Robust basic displacement plus a penalized, broad local residual."""
    empty = np.zeros(shape, float)
    accepted = [p for p in patches if p["training"]
                and abs(p.get("correction", p["shift"])) <= parameters.max_shift_mm
                and p.get("unambiguous_fraction", 1.) >= parameters.minimum_unambiguous_fraction
                and p.get("crossing_iqr_mm", 0.) <= parameters.maximum_crossing_iqr_mm
                and p.get("reliability", 1.) > 0]
    info = dict(accepted_fitting_patches=len(accepted), activated=False,
                baseline_mm=0., maximum_residual_mm=0., final_scale=1.)
    if not accepted:
        return empty.copy(), empty.copy(), empty.copy(), info
    targets = np.array([p.get("correction", p["shift"]) for p in accepted])
    reliability = np.array([p.get("reliability", 1.) for p in accepted])
    baseline = weighted_median(targets, reliability)
    q25 = weighted_quantile(targets, reliability, .25)
    q75 = weighted_quantile(targets, reliability, .75)
    info["baseline_mm"] = baseline
    info.update(weighted_target_q25_mm=q25, weighted_target_q75_mm=q75)
    if parameters.organ_bias_gate_mm is not None and (
            len(accepted) < parameters.minimum_training_patches
            or abs(baseline) <= parameters.organ_bias_gate_mm):
        return empty.copy(), empty.copy(), empty.copy(), info
    if parameters.organ_bias_gate_mm is not None and parameters.require_consistent_direction:
        if not (q25 > parameters.organ_bias_gate_mm or q75 < -parameters.organ_bias_gate_mm):
            info['activation_rejection'] = 'Central half of weighted targets does not lie beyond the deadband on one side'
            return empty.copy(), empty.copy(), empty.copy(), info
    points = np.array([np.mean([(c[0]*spacing[0], c[1]*spacing[1])
                              for c in p["columns"]], axis=0) for p in accepted])
    step = parameters.control_spacing_mm
    axes = [np.arange(np.floor(points[:, i].min()/step)*step,
                      np.ceil(points[:, i].max()/step)*step + step*.5, step) for i in (0, 1)]
    controls = np.array(np.meshgrid(*axes, indexing="ij")).reshape(2, -1).T
    basis = radial_basis(points, controls, parameters.residual_sigma_mm)
    weights = reliability / reliability.sum()
    coefficients = np.zeros(len(controls))
    for iteration in range(50):
        residual = targets - baseline - basis @ coefficients
        robust = np.minimum(1., parameters.huber_delta_mm / np.maximum(np.abs(residual), 1e-12))
        w = weights * robust
        lhs = basis.T @ (w[:, None] * basis) + np.eye(len(controls))*parameters.residual_penalty/len(controls)
        rhs = basis.T @ (w * (targets - baseline))
        updated = np.linalg.solve(lhs, rhs)
        if np.max(np.abs(updated - coefficients)) < 1e-7:
            coefficients = updated
            break
        coefficients = updated
    support_weights, heights = empty.copy(), empty.copy()
    for patch in accepted:
        for x, y, top, _ in patch["columns"]:
            support_weights[x, y] = patch.get("reliability", 1.)
            heights[x, y] = (top+.5)*spacing[2]*support_weights[x, y]
    sigma = parameters.spatial_sigma_mm / np.asarray(spacing[:2])
    denominator = gaussian_filter(support_weights, sigma, mode="constant")
    center = gaussian_filter(heights, sigma, mode="constant") / np.maximum(denominator, 1e-12)
    support = envelope_support(support_weights, spacing, parameters.support_distance_mm)
    coordinates = np.argwhere(support > 0)
    residual_field = empty.copy()
    for start in range(0, len(coordinates), 8192):
        indices = coordinates[start:start+8192]
        residual_field[tuple(indices.T)] = radial_basis(indices*spacing[:2], controls,
                                                       parameters.residual_sigma_mm) @ coefficients
    maximum = np.max(np.abs(residual_field))
    residual_scale = min(1., parameters.maximum_local_residual_mm / max(maximum, 1e-12))
    residual_field *= residual_scale
    shift = (baseline + residual_field)*support
    final_scale = min(1., parameters.max_shift_mm / max(np.max(np.abs(shift)), 1e-12))
    shift *= final_scale
    info.update(activated=True, controls=len(controls), irls_iterations=iteration+1,
                maximum_residual_mm=float(np.max(np.abs(residual_field))),
                residual_scale=float(residual_scale), final_scale=float(final_scale),
                weighted_training_rmse_mm=float(np.sqrt(np.sum(weights*(targets-(baseline+basis@coefficients*residual_scale)*final_scale)**2))))
    return shift, center, denominator*support, info


def fit_field(patches, shape, spacing, parameters):
    if parameters.field_model == "shape_regularized":
        return fit_field_shape(patches, shape, spacing, parameters)[:3]
    return fit_field_legacy(patches, shape, spacing, parameters)


def fit_field_legacy(patches, shape, spacing, parameters):
    """Smooth displacement using only prespecified training patches."""
    weights = np.zeros(shape, np.float64)
    numerator = weights.copy()
    heights = weights.copy()
    training = [p for p in patches if p["training"]]
    if parameters.organ_bias_gate_mm is not None:
        estimates = [p.get("correction", p["shift"]) for p in training]
        if len(estimates) < parameters.minimum_training_patches or abs(np.median(estimates)) <= parameters.organ_bias_gate_mm:
            return weights.copy(), weights.copy(), weights.copy()
    for patch in patches:
        estimate = patch.get("correction", patch["shift"])
        if not patch["training"] or abs(estimate) > parameters.max_shift_mm:
            continue  # Extreme estimates are rejected, not silently clipped.
        for x, y, top, _ in patch["columns"]:
            weights[x, y] = 1
            numerator[x, y] = estimate
            heights[x, y] = (top + .5) * spacing[2]
    if not np.any(weights):
        return np.zeros(shape), np.zeros(shape), np.zeros(shape)
    sigma = parameters.spatial_sigma_mm / np.asarray(spacing[:2])
    denominator = gaussian_filter(weights, sigma, mode="constant")
    shift = gaussian_filter(numerator, sigma, mode="constant") / np.maximum(denominator, 1e-12)
    center = gaussian_filter(heights, sigma, mode="constant") / np.maximum(denominator, 1e-12)
    distance = distance_transform_edt(weights == 0, sampling=spacing[:2])
    # Continuous lateral taper; only a finite neighbourhood can be changed.
    support = np.clip(1 - distance / parameters.support_distance_mm, 0, 1)
    shift *= support
    return shift, center, denominator * support


def inverse_si_coordinates(z, center, shift, band):
    """Inverse of z -> z + shift * max(1-|z-center|/band, 0)."""
    if np.any(np.abs(shift) >= band):
        raise ValueError("Displacement must be smaller than the taper band")
    lower, peak, upper = center - band, center + shift, center + band
    return np.where((z >= lower) & (z <= peak),
                    lower + (z - lower) * band / (band + shift),
                    np.where((z > peak) & (z <= upper),
                             center + (z - peak) * band / (band - shift), z))


def warp_labels(labels, shift, center, spacing_z, band):
    output = labels.copy()
    changed_columns = np.argwhere(np.abs(shift) > 1e-8)
    z = np.arange(labels.shape[2]) * spacing_z
    for x, y in changed_columns:
        source = inverse_si_coordinates(z, center[x, y], shift[x, y], band) / spacing_z
        indices = np.clip(np.floor(source + .5).astype(int), 0, labels.shape[2] - 1)
        output[x, y] = labels[x, y, indices]
    return output


def geometry(image, reference):
    if image.shape != reference.shape or not np.allclose(image.affine, reference.affine, atol=1e-5, rtol=0):
        raise ValueError("PET and every CT-grid mask must share shape and affine")
    matrix = reference.affine[:3, :3]
    if not np.allclose(matrix, np.diag(np.diag(matrix)), atol=1e-5) or matrix[2, 2] <= 0:
        raise ValueError("Only axis-aligned images with increasing superior z are supported")


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows for {path}")
    with open(path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(pet_path, mask_dir, output_dir, parameters=Parameters()):
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    if (parameters.training_parity not in (0, 1) or not (0 < parameters.max_shift_mm < parameters.band_mm)
            or parameters.adaptation_mode not in ("organ_edge", "interface_interval")
            or parameters.field_model not in ("shape_regularized", "legacy", "guarded_smooth")
            or min(parameters.control_spacing_mm, parameters.residual_sigma_mm,
                   parameters.residual_penalty, parameters.huber_delta_mm,
                   parameters.support_distance_mm) <= 0):
        raise ValueError("Invalid parameters")
    reference = nib.load(str(mask_dir / "liver.nii.gz"))
    pet_image = nib.load(str(pet_path))
    geometry(pet_image, reference)
    spacing = np.array(reference.header.get_zooms()[:3], float)
    pet = pet_image.get_fdata(dtype=np.float32)
    labels = np.zeros(reference.shape, np.uint8)
    inputs = {"pet": str(pet_path.resolve()), "pet_sha256": sha256(pet_path), "masks": {}}
    for index, name in enumerate(LABELS, 1):
        path = mask_dir / f"{name}.nii.gz"
        image = nib.load(str(path))
        geometry(image, reference)
        mask = np.asarray(image.dataobj) > 0
        if np.any(mask & (labels > 0)):
            raise ValueError(f"Overlapping source target masks: {name}")
        labels[mask] = index
        inputs["masks"][name] = {"path": str(path.resolve()), "sha256": sha256(path)}
    lung = labels >= 3
    all_patches, fields, field_details = {}, [], {}
    guard = None
    protected = np.zeros(labels.shape, bool)
    protection_inputs = {}
    if parameters.field_model == "guarded_smooth":
        from guarded_shape import fit_fields
        for path in sorted(mask_dir.glob("*.nii.gz")):
            name = path.name.removesuffix(".nii.gz")
            if name in LABELS or name.startswith("combined"):
                continue
            image = nib.load(str(path))
            geometry(image, reference)
            protected |= np.asarray(image.dataobj) > 0
            protection_inputs[name] = sha256(path)
        all_patches, fields, field_details, guard = fit_fields(
            pet, labels, protected, spacing, parameters)
    for index, name in ([] if guard is not None else enumerate(LABELS[:2], 1)):
        patches = get_patches(pet, labels == index, lung, spacing, parameters)
        all_patches[name] = patches
        if parameters.field_model == "shape_regularized":
            fitted = fit_field_shape(patches, labels.shape[:2], spacing, parameters)
            fields.append(fitted[:3])
            field_details[name] = fitted[3]
        else:
            fields.append(fit_field(patches, labels.shape[:2], spacing, parameters))
    del pet, lung
    # A single source coordinate is used for every target label, preventing
    # new target-target overlaps, including where the two organ supports meet.
    confidence = fields[0][2] + fields[1][2]
    shift = sum(f[0] * f[2] for f in fields) / np.maximum(confidence, 1e-12)
    center = sum(f[1] * f[2] for f in fields) / np.maximum(confidence, 1e-12)
    adapted = warp_labels(labels, shift, center, spacing[2], parameters.band_mm)
    # Do not assign target labels into other segmented organs. Preserve the
    # original target value there; report all vetoed assignments.
    for path in (sorted(mask_dir.glob("*.nii.gz"))
                 if guard is None and np.any(adapted != labels) else []):
        name = path.name.removesuffix(".nii.gz")
        if name in LABELS or name.startswith("combined"):
            continue
        image = nib.load(str(path))
        geometry(image, reference)
        protected |= np.asarray(image.dataobj) > 0
        protection_inputs[name] = sha256(path)
    veto = (adapted != labels) & (adapted > 0) & protected
    veto_count = int(veto.sum())
    adapted[veto] = labels[veto]
    output_dir.mkdir(parents=True)
    (output_dir / "masks").mkdir()
    volume_rows, patch_rows, profile_rows = [], [], []
    for index, name in enumerate(LABELS, 1):
        old, new = labels == index, adapted == index
        header = reference.header.copy()
        header.set_data_dtype(np.uint8)
        nib.save(nib.Nifti1Image(new.astype(np.uint8), reference.affine, header),
                 str(output_dir / "masks" / f"{name}.nii.gz"))
        n0, n1 = int(old.sum()), int(new.sum())
        volume_rows.append(dict(organ=name, original_voxels=n0, adapted_voxels=n1,
                                original_ml=n0 * np.prod(spacing) / 1000,
                                adapted_ml=n1 * np.prod(spacing) / 1000,
                                volume_change_percent=100 * (n1 - n0) / n0 if n0 else None,
                                changed_voxels=int(np.count_nonzero(old != new))))
        if name not in all_patches:
            continue
        for patch in all_patches[name]:
            deltas, profiles, new_gaps = [], [], []
            for x, y, top, profile in patch["columns"]:
                occupied = np.flatnonzero(new[x, y])
                if not occupied.size:
                    raise RuntimeError("An eligible organ column vanished")
                deltas.append((occupied[-1] - top) * spacing[2])
                profiles.append(profile)
                new_top = occupied[-1]
                lung_above = np.flatnonzero(adapted[x, y, new_top+1:new_top+8] >= 3)
                new_gaps.append(float(lung_above[0]*spacing[2]) if len(lung_above) else np.nan)
            # Fixed original columns, normalization and sampling positions.
            # Translate each column to its actual adapted binary-mask face.
            grid = (np.arange(-9, 10) - .5) * spacing[2]
            translated = np.array([np.interp(grid + d, grid, p, left=np.nan, right=np.nan)
                                   for p, d in zip(profiles, deltas)])
            common = np.all(np.isfinite(translated), axis=0)
            before = np.median(profiles, axis=0)
            after = np.full(19, np.nan)
            after[common] = np.median(translated[:, common], axis=0)
            # Interior crossings use only common support; extend tails with
            # endpoint values solely for the Gaussian boundary convention.
            valid = np.flatnonzero(common)
            if len(valid) < 9:
                raise RuntimeError("Insufficient common profile support")
            filled = np.interp(np.arange(19), valid, after[valid])
            after_cross = crossings(filled, spacing[2], parameters.profile_sigma_voxels)[.5]
            fold = "training" if patch["training"] else "held_out"
            key = f"{name}_{patch['key'][0]}_{patch['key'][1]}"
            patch_rows.append(dict(organ=name, patch=key, split=fold, columns=len(profiles),
                                   before_mm=patch["shift"], after_mm=after_cross,
                                   original_interface_gap_mm=patch["gap_mm"],
                                   adapted_interface_gap_mm=float(np.median(new_gaps)),
                                   before_interval_error_mm=float(interval_error(patch["shift"], 0., patch["gap_mm"])),
                                   after_interval_error_mm=float(interval_error(after_cross, 0., np.median(new_gaps))) if np.isfinite(new_gaps).all() else np.nan,
                                   median_applied_boundary_shift_mm=float(np.median(deltas)),
                                   width_before_mm=patch["width"]))
            for j in range(19):
                profile_rows.append(dict(organ=name, patch=key, split=fold,
                                         distance_mm=grid[j], before=before[j], after=after[j]))
    if guard is not None:
        from guarded_shape import verify_predictions
        verify_predictions(patch_rows, guard)
        (output_dir / "guard_report.json").write_text(json.dumps(guard, indent=2))
    np.savez_compressed(output_dir / "displacement_field.npz", shift_mm=shift, center_mm=center)
    write_csv(output_dir / "volume_qc.csv", volume_rows)
    write_csv(output_dir / "paired_patches.csv", patch_rows)
    write_csv(output_dir / "paired_profiles.csv", profile_rows)
    fit_qc = {}
    for name, patches in all_patches.items():
        training = [p for p in patches if p["training"]]
        estimates = [p["correction"] for p in training]
        fit_qc[name] = dict(training_patches=len(training),
                            median_training_correction_mm=float(np.median(estimates)) if estimates else None,
                            nonzero_field=bool(np.any(fields[LABELS.index(name)][0])))
    manifest = dict(parameters=asdict(parameters), inputs=inputs, organ_fit_qc=fit_qc,
                    field_details=field_details,
                    protected_mask_sha256=protection_inputs, protected_assignment_vetoes=veto_count,
                    minimum_forward_si_derivative=float(1 - np.max(np.abs(shift)) / parameters.band_mm),
                    source_sha256=sha256(Path(__file__)),
                    interpretation="Experimental PET-boundary adaptation; not anatomical ground truth")
    if guard is not None:
        import guarded_shape
        manifest["guarded_method"] = dict(
            source_sha256=sha256(Path(guarded_shape.__file__)),
            guard_scale=guard["scale"], maximum_patch_worsening_mm=1.5,
            support="CT lung-facing footprint, 10-mm Gaussian, 15-mm outward cosine taper")
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return patch_rows, volume_rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pet", type=Path, required=True, help="SUV PET already resampled to CT grid")
    parser.add_argument("--masks", type=Path, required=True, help="TotalSegmentator CT-grid masks")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("organ_edge", "interface_interval"), default="interface_interval")
    parser.add_argument("--maximum-interface-gap-mm", type=float, default=6.)
    parser.add_argument("--deadband-mm", type=float, default=1.5)
    parser.add_argument("--organ-bias-gate-mm", type=float, default=1.5)
    parser.add_argument("--field-model", choices=("guarded_smooth", "shape_regularized", "legacy"),
                        default="guarded_smooth")
    args = parser.parse_args()
    from overlap_gate import run_checked
    from run_adaptation_pipeline import adaptation_inventory
    run_checked(run, args.pet, args.masks, args.output, Parameters(adaptation_mode=args.mode,
        maximum_interface_gap_mm=args.maximum_interface_gap_mm, deadband_mm=args.deadband_mm,
        organ_bias_gate_mm=args.organ_bias_gate_mm, field_model=args.field_model),
        expected_names=adaptation_inventory())
