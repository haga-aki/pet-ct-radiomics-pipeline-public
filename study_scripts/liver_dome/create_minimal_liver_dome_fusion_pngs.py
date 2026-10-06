"""Minimal coronal CT/PET/liver-mask fusion images at the hepatic dome.

Study-specific visual QC script for the JJR technical note (Fig. 4).
See README.md in this directory for inputs.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter1d


# Set from the command line (see parse_args); no default data locations.
DATA_DIR: Path
PET_DIR: Path
OUTPUT_DIR: Path
CASES = [f"Case_{index:03d}" for index in range(1, 6)]

CT_WINDOW = (-250.0, 250.0)
PET_WINDOW = (0.0, 5.0)
X_HALF_WIDTH_MM = 110.0
Z_INFERIOR_MM = 70.0
Z_SUPERIOR_MM = 80.0
MASK_COLOR = "#00E5FF"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_coronal_plane(liver: np.ndarray, spacing: np.ndarray) -> int:
    coordinates = np.argwhere(liver)
    top_z = int(coordinates[:, 2].max())
    upper_depth = max(1, int(np.ceil(30.0 / spacing[2])))
    upper = liver[:, :, max(0, top_z - upper_depth) : top_z + 1]
    y_score = upper.sum(axis=(0, 2)).astype(float)
    y_score = gaussian_filter1d(y_score, sigma=max(1.0, 8.0 / spacing[1]))
    return int(np.argmax(y_score))


def crop_bounds(
    liver: np.ndarray, y_index: int, spacing: np.ndarray
) -> tuple[int, int, int, int, int, int]:
    plane = liver[:, y_index, :]
    coordinates = np.argwhere(plane)
    if coordinates.size == 0:
        raise RuntimeError("Selected coronal plane does not intersect liver mask")
    top_z = int(coordinates[:, 1].max())
    upper_depth = max(1, int(np.ceil(30.0 / spacing[2])))
    upper_coordinates = np.argwhere(plane[:, max(0, top_z - upper_depth) : top_z + 1])
    center_x = int(np.round(np.median(upper_coordinates[:, 0])))

    x_half = int(np.ceil(X_HALF_WIDTH_MM / spacing[0]))
    z_inferior = int(np.ceil(Z_INFERIOR_MM / spacing[2]))
    z_superior = int(np.ceil(Z_SUPERIOR_MM / spacing[2]))
    x_lo = max(0, center_x - x_half)
    x_hi = min(liver.shape[0], center_x + x_half + 1)
    z_lo = max(0, top_z - z_inferior)
    z_hi = min(liver.shape[2], top_z + z_superior + 1)
    return x_lo, x_hi, z_lo, z_hi, center_x, top_z


def load_case(case_id: str) -> dict[str, object]:
    case_dir = DATA_DIR / case_id
    ct_image = nib.load(case_dir / f"{case_id}_CT.nii.gz")
    pet_image = nib.load(
        PET_DIR / case_id / f"{case_id}_PET_SUV_corrected.nii.gz"
    )
    liver_image = nib.load(case_dir / f"{case_id}_liver.nii.gz")
    if ct_image.shape != pet_image.shape or ct_image.shape != liver_image.shape:
        raise RuntimeError(f"{case_id}: CT/PET/liver shape mismatch")
    if not np.allclose(ct_image.affine, pet_image.affine) or not np.allclose(
        ct_image.affine, liver_image.affine
    ):
        raise RuntimeError(f"{case_id}: CT/PET/liver affine mismatch")
    if nib.aff2axcodes(ct_image.affine) != ("L", "A", "S"):
        raise RuntimeError(f"{case_id}: unexpected orientation {nib.aff2axcodes(ct_image.affine)}")

    ct = np.asarray(ct_image.dataobj, dtype=np.float32)
    pet = np.asarray(pet_image.dataobj, dtype=np.float32)
    liver = np.asarray(liver_image.dataobj) > 0
    spacing = np.asarray(ct_image.header.get_zooms()[:3], dtype=float)
    y_index = select_coronal_plane(liver, spacing)
    x_lo, x_hi, z_lo, z_hi, center_x, top_z = crop_bounds(liver, y_index, spacing)

    x_centers = (np.arange(x_lo, x_hi) - center_x) * spacing[0]
    z_centers = (np.arange(z_lo, z_hi) - top_z) * spacing[2]
    extent = [
        float(x_centers[0] - spacing[0] / 2),
        float(x_centers[-1] + spacing[0] / 2),
        float(z_centers[0] - spacing[2] / 2),
        float(z_centers[-1] + spacing[2] / 2),
    ]
    y_world = float(nib.affines.apply_affine(ct_image.affine, [0, y_index, 0])[1])
    top_world_z = float(nib.affines.apply_affine(ct_image.affine, [0, 0, top_z])[2])
    return {
        "case_id": case_id,
        "ct": ct[x_lo:x_hi, y_index, z_lo:z_hi].T,
        "pet": pet[x_lo:x_hi, y_index, z_lo:z_hi].T,
        "liver": liver[x_lo:x_hi, y_index, z_lo:z_hi].T,
        "extent": extent,
        "spacing": spacing,
        "shape": ct_image.shape,
        "y_index": y_index,
        "y_world_mm": y_world,
        "center_x_index": center_x,
        "top_z_index": top_z,
        "top_world_z_mm": top_world_z,
        "crop_indices": (x_lo, x_hi, z_lo, z_hi),
    }


def draw_fusion(ax, case: dict[str, object], show_case_title: bool = True) -> None:
    ct = case["ct"]
    pet = case["pet"]
    liver = case["liver"]
    extent = case["extent"]
    ax.imshow(
        ct,
        cmap="gray",
        vmin=CT_WINDOW[0],
        vmax=CT_WINDOW[1],
        origin="lower",
        extent=extent,
        interpolation="bilinear",
        aspect="equal",
    )

    norm = Normalize(*PET_WINDOW, clip=True)
    pet_rgba = plt.get_cmap("magma")(norm(pet))
    pet_alpha = 0.10 + 0.40 * np.clip(pet / PET_WINDOW[1], 0, 1)
    pet_alpha[pet < 0.5] = 0
    pet_rgba[..., 3] = pet_alpha
    ax.imshow(
        pet_rgba,
        origin="lower",
        extent=extent,
        interpolation="bilinear",
        aspect="equal",
    )
    ax.contour(
        liver.astype(float),
        levels=[0.5],
        colors=[MASK_COLOR],
        linewidths=1.5,
        origin="lower",
        extent=extent,
    )

    if show_case_title:
        ax.set_title(f"Case {int(case['case_id'].split('_')[-1])}", loc="left", fontweight="bold", fontsize=12)
    ax.set_xlabel("Right–left position relative to dome center (mm)")
    ax.set_ylabel("Superior–inferior position relative to CT-mask dome (mm)")
    ax.text(0.01, 0.985, "R", transform=ax.transAxes, va="top", ha="left", color="white", fontweight="bold")
    ax.text(0.99, 0.985, "L", transform=ax.transAxes, va="top", ha="right", color="white", fontweight="bold")
    ax.text(
        0.015,
        0.02,
        "CT −250 to 250 HU  |  PET 0–5 SUV  |  cyan: CT-derived liver mask",
        transform=ax.transAxes,
        fontsize=7.5,
        color="white",
        ha="left",
        va="bottom",
        bbox={"facecolor": "black", "alpha": 0.55, "edgecolor": "none", "pad": 2.5},
    )
    scale_y = extent[2] + 8
    scale_x_end = extent[1] - 8
    ax.plot([scale_x_end - 20, scale_x_end], [scale_y, scale_y], color="white", linewidth=3)
    ax.text(scale_x_end - 10, scale_y + 3, "20 mm", color="white", ha="center", va="bottom", fontsize=8)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)


def write_individual(case: dict[str, object]) -> Path:
    fig, ax = plt.subplots(figsize=(6.2, 5.4), constrained_layout=True)
    draw_fusion(ax, case)
    colorbar = fig.colorbar(
        ScalarMappable(norm=Normalize(*PET_WINDOW), cmap="magma"),
        ax=ax,
        fraction=0.045,
        pad=0.025,
    )
    colorbar.set_label("PET SUV", fontsize=9)
    colorbar.ax.tick_params(labelsize=8)
    output = OUTPUT_DIR / f"{case['case_id']}_liver_dome_coronal_CT_PET_mask_fusion.png"
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output


def write_contact_sheet(cases: list[dict[str, object]]) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 9.6), constrained_layout=True)
    for ax, case in zip(axes.flat, cases):
        draw_fusion(ax, case)
    axes.flat[-1].axis("off")
    colorbar = fig.colorbar(
        ScalarMappable(norm=Normalize(*PET_WINDOW), cmap="magma"),
        ax=list(axes.flat[:-1]),
        fraction=0.018,
        pad=0.015,
    )
    colorbar.set_label("PET SUV")
    fig.suptitle(
        "Minimal visual QC: SUV-corrected coronal CT/PET fusion at the CT-derived liver dome",
        fontsize=16,
        fontweight="bold",
    )
    output = OUTPUT_DIR / "all_cases_liver_dome_coronal_CT_PET_mask_fusion_contact_sheet.png"
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cases = [load_case(case_id) for case_id in CASES]
    output_rows = []
    for case in cases:
        output = write_individual(case)
        x_lo, x_hi, z_lo, z_hi = case["crop_indices"]
        output_rows.append(
            {
                "CaseID": case["case_id"],
                "PNG": output.name,
                "sha256": sha256(output),
                "source_shape": "x".join(str(value) for value in case["shape"]),
                "source_spacing_mm": "x".join(f"{value:.6g}" for value in case["spacing"]),
                "selected_coronal_y_index_zero_based": case["y_index"],
                "selected_coronal_y_world_mm": f"{case['y_world_mm']:.3f}",
                "CT_mask_dome_z_index_zero_based": case["top_z_index"],
                "CT_mask_dome_z_world_mm": f"{case['top_world_z_mm']:.3f}",
                "crop_x_indices_half_open": f"{x_lo}:{x_hi}",
                "crop_z_indices_half_open": f"{z_lo}:{z_hi}",
                "CT_window_HU": f"{CT_WINDOW[0]:.0f}:{CT_WINDOW[1]:.0f}",
                "PET_window_SUV": f"{PET_WINDOW[0]:.0f}:{PET_WINDOW[1]:.0f}",
            }
        )
    contact_sheet = write_contact_sheet(cases)
    manifest = OUTPUT_DIR / "visual_qc_png_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    print(contact_sheet)
    for row in output_rows:
        print(row["CaseID"], row["PNG"], row["sha256"])


def parse_args(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True,
                        help="Directory with <case>/<case>_CT.nii.gz and _liver.nii.gz files")
    parser.add_argument("--pet", type=Path, required=True,
                        help="Directory with <case>/<case>_PET_SUV_corrected.nii.gz (CT-grid PET SUV copy)")
    parser.add_argument("--out", type=Path, required=True, help="Output directory")
    parser.add_argument("--cases", nargs="+", default=CASES, help="Case identifiers")
    return parser.parse_args(argv)


if __name__ == "__main__":
    _args = parse_args()
    DATA_DIR, PET_DIR, OUTPUT_DIR, CASES = _args.inputs, _args.pet, _args.out, list(_args.cases)
    main()
