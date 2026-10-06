"""Run-scoped technical QC; never implies anatomical or clinical acceptance."""

import csv
import math
import os
import json
import platform
import hashlib
from importlib.metadata import version, PackageNotFoundError
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np


def geometry_metrics(image, mask, tolerance=1e-5):
    """Compare SimpleITK geometries in physical units; directions are unitless."""
    size_match = image.GetSize() == mask.GetSize()
    deltas = {}
    for name, getter in (("spacing", "GetSpacing"), ("origin", "GetOrigin"),
                         ("direction", "GetDirection")):
        a = np.asarray(getattr(image, getter)(), dtype=float)
        b = np.asarray(getattr(mask, getter)(), dtype=float)
        deltas[name + "_max_abs_diff"] = (
            float(np.max(np.abs(a - b))) if a.shape == b.shape else float("inf")
        )
    return {"geometry_status": "matched" if size_match and all(
        math.isfinite(v) and v <= tolerance for v in deltas.values()
    ) else "mismatch", "size_match": size_match, **deltas}


def nonfinite_features(result):
    return sorted(k for k, v in result.items() if k.startswith("original_")
                  and not np.all(np.isfinite(np.asarray(v, dtype=float))))


class QCReport:
    """Checkpoint separate files per invocation, retaining previous run evidence."""

    def __init__(self, output_dir):
        self.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
        self.directory = Path(output_dir) / "qc" / self.run_id
        self.directory.mkdir(parents=True, exist_ok=True)
        self.organs = []
        self.stages = []
        self.cases = []
        versions = {'python': platform.python_version()}
        for package in ('numpy', 'pydicom', 'SimpleITK', 'nibabel', 'pyradiomics',
                        'TotalSegmentator', 'dicom2nifti', 'torch', 'pandas', 'PyYAML'):
            try:
                versions[package] = version(package)
            except PackageNotFoundError:
                versions[package] = 'not_installed'
        (self.directory / 'runtime_versions.json').write_text(json.dumps(versions, indent=2), encoding='utf-8')
        self.flush()

    def provenance(self, config, root):
        """Link result rows to exact code/config without exporting DICOM IDs."""
        source = Path(__file__).resolve().parent
        digest = hashlib.sha256()
        for name in ('run_pipeline.py', 'qc_reporting.py', 'mask_inventory.py',
                     'suv_converter.py', 'suv_timing.py', 'radiomics_grid_policy.py'):
            digest.update(name.encode())
            digest.update((source / name).read_bytes())
        params = Path(config.get('radiomics', {}).get('params_file', 'params.yaml'))
        if not params.is_absolute():
            params = Path(root) / params
        row = dict(RunID=self.run_id, PipelineSHA256=digest.hexdigest(),
                   ConfigurationSHA256=hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
                   ParametersSHA256=hashlib.sha256(params.read_bytes()).hexdigest() if params.is_file() else 'unavailable')
        (self.directory / 'provenance.json').write_text(json.dumps(row, indent=2), encoding='utf-8')
        return row

    def start_case(self, case):
        if case not in self.cases:
            self.cases.append(case)
        self.stage(case, "case", "started")

    def stage(self, case, stage, status, reason="", modality="", artifact=""):
        self.stages.append(dict(case_id=case, modality=modality, stage=stage,
                                status=status, reason=reason, artifact=str(artifact)))
        self.flush()

    def organ(self, case, modality, organ, status, reason="", **metrics):
        self.organs.append(dict(case_id=case, modality=modality, organ=organ,
                                status=status, reason=reason, **metrics))
        self.flush()

    def summary(self):
        rows = []
        for case in self.cases:
            organs = [r for r in self.organs if r["case_id"] == case]
            stages = [r for r in self.stages if r["case_id"] == case]
            problems = [r for r in organs if r["status"] != "extracted"]
            failed = any(r["status"] == "failed" for r in stages)
            complete = any(r["stage"] == "case" and r["status"] == "finished" for r in stages)
            extracted = sum(r["status"] in ("extracted", "nonfinite_features") for r in organs)
            rows.append(dict(case_id=case, run_status="finished" if complete else "incomplete",
                             technical_status="incomplete" if not complete else (
                                 "attention_required" if failed or problems or not extracted
                                 else "checks_completed"),
                             organ_records=len(organs), extracted_rows=extracted,
                             attention_records=len(problems),
                             visual_review="pending",
                             anatomical_acceptance="not_assessed"))
        return rows

    def flush(self):
        for name, rows, base in (("organ_qc", self.organs, ["case_id", "modality", "organ", "status", "reason"]),
                                 ("stage_qc", self.stages, ["case_id", "modality", "stage", "status", "reason", "artifact"]),
                                 ("case_qc", self.summary(), ["case_id", "run_status", "technical_status", "visual_review"])):
            fields = ["run_id"] + list(dict.fromkeys(base + [k for row in rows for k in row]))
            target = self.directory / (name + ".csv")
            temporary = target.with_suffix(".tmp")
            with temporary.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(dict(run_id=self.run_id, **row) for row in rows)
            os.replace(temporary, target)
