"""Hepatic-dome PET transition profiles relative to the CT-derived liver boundary.

Study-specific analysis script for the JJR technical note (Figs. 5 and 6).
See README.md in this directory for inputs and parameters.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter1d


# Set from the command line (see parse_args); no default data locations.
DATA_DIR: Path
PET_DIR: Path
OUT_DIR: Path
CASES = [f"Case_{index:03d}" for index in range(1, 6)]
OFFSETS = np.arange(-9, 10)
PATCH_SIZE_MM = 10.0
MIN_PATCH_COLUMNS = 20


def crossing_mm(profile: np.ndarray, spacing: float, threshold: float = 0.5) -> float:
    """Return nearest descending threshold crossing; x=0 is the CT mask surface."""
    x_mm = (OFFSETS - 0.5) * spacing
    smooth = gaussian_filter1d(np.asarray(profile, dtype=float), sigma=0.7, mode="nearest")
    candidates = []
    for index in range(len(smooth) - 1):
        left, right = smooth[index], smooth[index + 1]
        if left >= threshold and right < threshold and left != right:
            fraction = (threshold - left) / (right - left)
            candidates.append(x_mm[index] + fraction * (x_mm[index + 1] - x_mm[index]))
    return min(candidates, key=abs) if candidates else float("nan")


def measure_case(
    case_id: str,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    case_dir = DATA_DIR / case_id
    pet_image = nib.load(
        str(PET_DIR / case_id / f"{case_id}_PET_SUV_corrected.nii.gz")
    )
    liver_image = nib.load(str(case_dir / f"{case_id}_liver.nii.gz"))
    lung_image = nib.load(str(case_dir / f"{case_id}_lung_whole.nii.gz"))
    if nib.aff2axcodes(pet_image.affine)[2] != "S":
        raise RuntimeError(f"{case_id}: expected increasing z to be superior")

    pet = pet_image.get_fdata(dtype=np.float32)
    liver = liver_image.get_fdata(dtype=np.float32) > 0.5
    lung = lung_image.get_fdata(dtype=np.float32) > 0.5
    spacing = float(pet_image.header.get_zooms()[2])
    nx, _, nz = liver.shape

    profiles = []
    column_coordinates = []
    column_top_indices = []
    inside_values = []
    outside_values = []
    for x_index in range(nx):
        for y_index in np.flatnonzero(liver[x_index].any(axis=1)):
            z_indices = np.flatnonzero(liver[x_index, y_index])
            if z_indices.size < 7:
                continue
            top = int(z_indices[-1])
            if top < 10 or top + 10 >= nz:
                continue
            if np.count_nonzero(lung[x_index, y_index, top + 1 : top + 8]) < 2:
                continue
            profile = pet[x_index, y_index, top + OFFSETS].astype(float)
            if not np.all(np.isfinite(profile)):
                continue
            inside = float(np.median(profile[2:7]))
            outside = float(np.median(profile[11:17]))
            contrast = inside - outside
            if inside < 1.0 or contrast < 0.35:
                continue
            profiles.append((profile - outside) / contrast)
            column_coordinates.append((x_index, y_index))
            column_top_indices.append(top)
            inside_values.append(inside)
            outside_values.append(outside)

    profile_array = np.asarray(profiles)
    median_profile = np.nanmedian(profile_array, axis=0)
    q25_profile = np.nanpercentile(profile_array, 25, axis=0)
    q75_profile = np.nanpercentile(profile_array, 75, axis=0)
    smooth_profile = gaussian_filter1d(median_profile, sigma=0.7, mode="nearest")
    smooth_q25 = gaussian_filter1d(q25_profile, sigma=0.7, mode="nearest")
    smooth_q75 = gaussian_filter1d(q75_profile, sigma=0.7, mode="nearest")
    crossings = {threshold: crossing_mm(median_profile, spacing, threshold) for threshold in (0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9)}
    result = {
        "CaseID": case_id,
        "eligible_dome_columns": len(profiles),
        "liver_inside_SUV_median": float(np.median(inside_values)),
        "superior_lung_SUV_median": float(np.median(outside_values)),
        "PET50_minus_CT_boundary_mm": crossings[0.5],
        "PET40_minus_CT_boundary_mm": crossings[0.4],
        "PET60_minus_CT_boundary_mm": crossings[0.6],
        "transition_width_20_80_mm": crossings[0.2] - crossings[0.8],
        "transition_width_10_90_mm": crossings[0.1] - crossings[0.9],
    }
    x_mm = (OFFSETS - 0.5) * spacing
    profile_rows = [
        {
            "CaseID": case_id,
            "distance_from_CT_liver_mask_boundary_mm": float(distance),
            "median_normalized_PET": float(raw),
            "smoothed_median_normalized_PET": float(smooth),
            "normalized_PET_q25": float(q25),
            "normalized_PET_q75": float(q75),
            "smoothed_normalized_PET_q25": float(smooth_low),
            "smoothed_normalized_PET_q75": float(smooth_high),
        }
        for distance, raw, smooth, q25, q75, smooth_low, smooth_high in zip(
            x_mm,
            median_profile,
            smooth_profile,
            q25_profile,
            q75_profile,
            smooth_q25,
            smooth_q75,
        )
    ]

    coordinates = np.asarray(column_coordinates, dtype=int)
    top_indices = np.asarray(column_top_indices, dtype=int)
    inplane_spacing = np.asarray(pet_image.header.get_zooms()[:2], dtype=float)
    relative_xy_mm = (coordinates - coordinates.min(axis=0)) * inplane_spacing
    patch_indices = np.floor(relative_xy_mm / PATCH_SIZE_MM).astype(int)
    patch_groups: dict[tuple[int, int], list[int]] = {}
    for column_index, patch_index in enumerate(patch_indices):
        patch_groups.setdefault(tuple(int(value) for value in patch_index), []).append(column_index)

    patch_rows = []
    patch_number = 0
    for patch_key, member_indices_list in sorted(patch_groups.items()):
        member_indices = np.asarray(member_indices_list, dtype=int)
        if len(member_indices) < MIN_PATCH_COLUMNS:
            continue
        patch_profile = np.nanmedian(profile_array[member_indices], axis=0)
        patch_crossings = {
            threshold: crossing_mm(patch_profile, spacing, threshold)
            for threshold in (0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9)
        }
        ordered = [patch_crossings[0.8], patch_crossings[0.5], patch_crossings[0.2]]
        if not np.all(np.isfinite(ordered)) or not (ordered[0] <= ordered[1] <= ordered[2]):
            continue
        width_20_80 = patch_crossings[0.2] - patch_crossings[0.8]
        width_10_90 = patch_crossings[0.1] - patch_crossings[0.9]
        if not np.isfinite(width_10_90) or width_20_80 <= 0 or width_10_90 <= 0:
            continue

        patch_number += 1
        member_xyz = np.column_stack(
            [coordinates[member_indices], top_indices[member_indices]]
        )
        center_ijk = np.median(member_xyz, axis=0)
        center_world = nib.affines.apply_affine(pet_image.affine, center_ijk)
        patch_rows.append(
            {
                "CaseID": case_id,
                "PatchID": f"{case_id}_P{patch_number:03d}",
                "patch_x_index": patch_key[0],
                "patch_y_index": patch_key[1],
                "eligible_columns": int(len(member_indices)),
                "projected_area_mm2": float(len(member_indices) * np.prod(inplane_spacing)),
                "center_world_x_mm": float(center_world[0]),
                "center_world_y_mm": float(center_world[1]),
                "center_world_z_mm": float(center_world[2]),
                "PET50_minus_CT_boundary_mm": patch_crossings[0.5],
                "PET40_minus_CT_boundary_mm": patch_crossings[0.4],
                "PET60_minus_CT_boundary_mm": patch_crossings[0.6],
                "transition_width_20_80_mm": width_20_80,
                "transition_width_10_90_mm": width_10_90,
            }
        )

    if not patch_rows:
        raise RuntimeError(f"{case_id}: no valid projected hepatic-dome patches")
    patch_shift = np.asarray([row["PET50_minus_CT_boundary_mm"] for row in patch_rows], dtype=float)
    patch_width = np.asarray([row["transition_width_20_80_mm"] for row in patch_rows], dtype=float)
    represented_area = float(sum(row["projected_area_mm2"] for row in patch_rows))
    eligible_area = float(len(profiles) * np.prod(inplane_spacing))
    patch_summary = {
        "CaseID": case_id,
        "patch_size_mm": PATCH_SIZE_MM,
        "minimum_eligible_columns_per_patch": MIN_PATCH_COLUMNS,
        "valid_patch_count": len(patch_rows),
        "eligible_projected_area_mm2": eligible_area,
        "represented_projected_area_mm2": represented_area,
        "represented_area_fraction": represented_area / eligible_area,
        "patch_PET50_median_mm": float(np.median(patch_shift)),
        "patch_PET50_q25_mm": float(np.percentile(patch_shift, 25)),
        "patch_PET50_q75_mm": float(np.percentile(patch_shift, 75)),
        "patch_PET50_min_mm": float(np.min(patch_shift)),
        "patch_PET50_max_mm": float(np.max(patch_shift)),
        "patch_width_20_80_median_mm": float(np.median(patch_width)),
        "patch_width_20_80_q25_mm": float(np.percentile(patch_width, 25)),
        "patch_width_20_80_q75_mm": float(np.percentile(patch_width, 75)),
        "patch_width_20_80_min_mm": float(np.min(patch_width)),
        "patch_width_20_80_max_mm": float(np.max(patch_width)),
    }
    return result, profile_rows, patch_rows, patch_summary


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(metrics: list[dict[str, object]], profiles: list[dict[str, object]]) -> Path:
    colors = ["#0072B2", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.4), constrained_layout=True)

    profile_ax = axes[0]
    for color, case_id in zip(colors, CASES):
        case_rows = [row for row in profiles if row["CaseID"] == case_id]
        x_mm = [row["distance_from_CT_liver_mask_boundary_mm"] for row in case_rows]
        y_value = [row["smoothed_median_normalized_PET"] for row in case_rows]
        profile_ax.fill_between(x_mm,
            [row['smoothed_normalized_PET_q25'] for row in case_rows],
            [row['smoothed_normalized_PET_q75'] for row in case_rows],
            color=color, alpha=0.10, linewidth=0)
        profile_ax.plot(x_mm, y_value, color=color, linewidth=2.0, label=f"Case {int(case_id.split('_')[-1])}")
    profile_ax.axvline(0, color="0.25", linewidth=1.2, linestyle="--")
    profile_ax.axhline(0.5, color="0.65", linewidth=1.0, linestyle=":")
    profile_ax.set_xlim(-24, 24)
    profile_ax.set_ylim(-0.1, 1.25)
    profile_ax.set_xlabel("Distance from CT-derived liver-mask boundary (mm)\nInferior  ←   →  Superior")
    profile_ax.set_ylabel("Normalized PET uptake")
    profile_ax.set_title("A  Liver-to-lung PET transition", loc="left", fontweight="bold")
    profile_ax.legend(frameon=False, fontsize=9, ncol=1)

    cases = [row["CaseID"] for row in metrics]
    y_pos = np.arange(len(cases))
    shift = np.asarray([row["PET50_minus_CT_boundary_mm"] for row in metrics], dtype=float)
    shift_ax = axes[1]
    shift_ax.axvspan(-3, 3, color="0.92", zorder=0)
    shift_ax.axvline(0, color="0.25", linewidth=1.2)
    for index, (color, value) in enumerate(zip(colors, shift)):
        shift_ax.plot([0, value], [index, index], color=color, linewidth=2)
        shift_ax.scatter([value], [index], color=color, s=65, zorder=3)
        shift_ax.text(value + (0.35 if value >= 0 else -0.35), index, f"{value:+.1f}", va="center", ha="left" if value >= 0 else "right", fontsize=9)
    shift_ax.set_yticks(y_pos, [f"Case {int(c.split('_')[-1])}" for c in cases])
    shift_ax.set_xlim(-7, 5)
    shift_ax.invert_yaxis()
    shift_ax.set_xlabel("PET 50% edge − CT-derived mask boundary (mm)\nCaudal  ←   →  Cranial")
    shift_ax.set_title("B  Apparent boundary displacement", loc="left", fontweight="bold")
    shift_ax.text(
        0.02,
        0.02,
        "Gray band: ±3 mm source-slice interval",
        transform=shift_ax.transAxes,
        color="0.35",
        fontsize=8,
        va="bottom",
    )

    widths = np.asarray([row["transition_width_20_80_mm"] for row in metrics], dtype=float)
    width_ax = axes[2]
    for index, (color, value) in enumerate(zip(colors, widths)):
        width_ax.barh(index, value, color=color, alpha=0.85, height=0.55)
        width_ax.text(value + 0.18, index, f"{value:.1f}", va="center", fontsize=9)
    width_ax.set_yticks(y_pos, [f"Case {int(c.split('_')[-1])}" for c in cases])
    width_ax.set_xlim(0, 14)
    width_ax.invert_yaxis()
    width_ax.set_xlabel("PET 20–80% transition width (mm)")
    width_ax.set_title("C  PET edge transition width", loc="left", fontweight="bold")

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", color="0.88", linewidth=0.7, zorder=0)
    fig.suptitle("PET uptake transition across the CT-derived liver-mask boundary", fontsize=15, fontweight="bold")
    fig.text(
        0.5,
        -0.015,
        "Profile-derived estimates from native 3-mm superior–inferior sampling; subvoxel values reflect interpolation, not independent spatial resolution.",
        ha="center",
        va="top",
        fontsize=8,
        color="0.35",
    )
    output = OUT_DIR / "liver_pet_edge_summary_publication.png"
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output


def plot_surface_distribution(
    profiles: list[dict[str, object]], patch_rows: list[dict[str, object]]
) -> Path:
    colors = ["#0072B2", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]
    fig, axes = plt.subplots(1, 3, figsize=(17.2, 5.7), constrained_layout=True)

    profile_ax = axes[0]
    for color, case_id in zip(colors, CASES):
        case_rows = [row for row in profiles if row["CaseID"] == case_id]
        x_mm = np.asarray([row["distance_from_CT_liver_mask_boundary_mm"] for row in case_rows])
        median = np.asarray([row["smoothed_median_normalized_PET"] for row in case_rows])
        q25 = np.asarray([row["smoothed_normalized_PET_q25"] for row in case_rows])
        q75 = np.asarray([row["smoothed_normalized_PET_q75"] for row in case_rows])
        profile_ax.fill_between(x_mm, q25, q75, color=color, alpha=0.10, linewidth=0)
        profile_ax.plot(x_mm, median, color=color, linewidth=2.0, label=f"Case {int(case_id.split('_')[-1])}")
    profile_ax.axvline(0, color="0.25", linewidth=1.2, linestyle="--")
    profile_ax.axhline(0.5, color="0.65", linewidth=1.0, linestyle=":")
    profile_ax.set_xlim(-24, 24)
    profile_ax.set_ylim(-0.1, 1.25)
    profile_ax.set_xlabel("Distance from CT-derived liver-mask boundary (mm)\nInferior  ←   →  Superior")
    profile_ax.set_ylabel("Normalized PET uptake")
    profile_ax.set_title("A  Surface-column profiles", loc="left", fontweight="bold")
    profile_ax.legend(frameon=False, fontsize=9)

    rng = np.random.default_rng(20260802)
    y_pos = np.arange(len(CASES))

    def distribution_panel(ax, field, title, xlabel, shade_slice_interval=False):
        all_values = []
        for case_id in CASES:
            values = np.asarray(
                [row[field] for row in patch_rows if row["CaseID"] == case_id], dtype=float
            )
            all_values.extend(values.tolist())
        all_values = np.asarray(all_values)
        if shade_slice_interval:
            ax.axvspan(-3, 3, color="0.92", zorder=0)
            ax.axvline(0, color="0.25", linewidth=1.1, zorder=1)

        for index, (color, case_id) in enumerate(zip(colors, CASES)):
            values = np.asarray(
                [row[field] for row in patch_rows if row["CaseID"] == case_id], dtype=float
            )
            jitter = rng.uniform(-0.17, 0.17, size=len(values))
            ax.scatter(values, index + jitter, color=color, alpha=0.35, s=15, linewidth=0, zorder=2)
            ax.boxplot(
                values,
                positions=[index],
                vert=False,
                widths=0.48,
                whis=(10, 90),
                showfliers=False,
                manage_ticks=False,
                patch_artist=True,
                boxprops={"facecolor": color, "edgecolor": color, "alpha": 0.22},
                whiskerprops={"color": color, "linewidth": 1.3},
                capprops={"color": color, "linewidth": 1.3},
                medianprops={"color": "0.12", "linewidth": 1.8},
            )
            median = float(np.median(values))
            ax.scatter([median], [index], marker="D", s=34, color=color, edgecolor="white", linewidth=0.7, zorder=4)

        data_min = float(np.min(all_values))
        data_max = float(np.max(all_values))
        if shade_slice_interval:
            span = max(data_max, 3) - min(data_min, -3)
            ax.set_xlim(np.floor(min(data_min, -3) - 0.08 * span), np.ceil(max(data_max, 3) + 0.08 * span))
        else:
            ax.set_xlim(0, np.ceil(data_max * 1.08))
        ax.set_yticks(y_pos, [f"Case {int(c.split('_')[-1])}" for c in CASES])
        ax.invert_yaxis()
        ax.set_xlabel(xlabel)
        ax.set_title(title, loc="left", fontweight="bold")

    distribution_panel(
        axes[1],
        "PET50_minus_CT_boundary_mm",
        "B  Local apparent displacement",
        "Patch PET 50% edge − CT-derived mask boundary (mm)\nCaudal  ←   →  Cranial",
        shade_slice_interval=True,
    )
    axes[1].text(
        0.02,
        0.02,
        "Gray band: ±3 mm source-slice interval",
        transform=axes[1].transAxes,
        color="0.35",
        fontsize=8,
        va="bottom",
    )
    distribution_panel(
        axes[2],
        "transition_width_20_80_mm",
        "C  Local PET transition width",
        "Patch PET 20–80% transition width (mm)",
    )

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", color="0.88", linewidth=0.7, zorder=0)
    fig.suptitle("Surface distribution of PET liver-dome transition metrics", fontsize=15, fontweight="bold")
    fig.text(
        0.5,
        -0.015,
        "Lines/bands in A: median and IQR across eligible surface columns. Points in B–C: valid 10 × 10-mm projected patches (≥20 columns); boxes: IQR, whiskers: 10th–90th percentiles, diamonds: medians. Native superior–inferior sampling: 3 mm.",
        ha="center",
        va="top",
        fontsize=8,
        color="0.35",
    )
    output = OUT_DIR / "liver_pet_edge_surface_patch_distribution.png"
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics = []
    profiles = []
    patch_metrics = []
    patch_summaries = []
    for case_id in CASES:
        result, case_profiles, case_patch_metrics, case_patch_summary = measure_case(case_id)
        metrics.append(result)
        profiles.extend(case_profiles)
        patch_metrics.extend(case_patch_metrics)
        patch_summaries.append(case_patch_summary)
    write_csv(OUT_DIR / "liver_pet_edge_metrics.csv", metrics)
    write_csv(OUT_DIR / "liver_pet_edge_profiles.csv", profiles)
    write_csv(OUT_DIR / "liver_pet_edge_patch_metrics.csv", patch_metrics)
    write_csv(OUT_DIR / "liver_pet_edge_patch_summary.csv", patch_summaries)
    output = plot_summary(metrics, profiles)
    surface_output = plot_surface_distribution(profiles, patch_metrics)
    for row in metrics:
        print(
            f"{row['CaseID']}: shift={row['PET50_minus_CT_boundary_mm']:+.3f} mm, "
            f"W20-80={row['transition_width_20_80_mm']:.3f} mm"
        )
    print(output)
    for row in patch_summaries:
        print(
            f"{row['CaseID']}: patches={row['valid_patch_count']}, "
            f"patch shift median={row['patch_PET50_median_mm']:+.3f} "
            f"[{row['patch_PET50_q25_mm']:+.3f}, {row['patch_PET50_q75_mm']:+.3f}] mm, "
            f"patch W20-80 median={row['patch_width_20_80_median_mm']:.3f} "
            f"[{row['patch_width_20_80_q25_mm']:.3f}, {row['patch_width_20_80_q75_mm']:.3f}] mm"
        )
    print(surface_output)


def parse_args(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True,
                        help="Directory with <case>/<case>_liver.nii.gz and _lung_whole.nii.gz files")
    parser.add_argument("--pet", type=Path, required=True,
                        help="Directory with <case>/<case>_PET_SUV_corrected.nii.gz (CT-grid PET SUV copy)")
    parser.add_argument("--out", type=Path, required=True, help="Output directory")
    parser.add_argument("--cases", nargs="+", default=CASES, help="Case identifiers")
    return parser.parse_args(argv)


if __name__ == "__main__":
    _args = parse_args()
    DATA_DIR, PET_DIR, OUT_DIR, CASES = _args.inputs, _args.pet, _args.out, list(_args.cases)
    main()
