"""Synthetic images and simulated failures; no patient data or model downloads."""
import ast
import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import SimpleITK as sitk

from qc_reporting import QCReport, geometry_metrics, nonfinite_features
from radiomics_grid_policy import merge_pet_and_shape_results


SOURCE = Path(__file__).resolve().parents[1] / 'run_pipeline.py'


def load_function(name, namespace):
    # Isolate orchestration from optional GPU/segmentation package imports.
    tree = ast.parse(SOURCE.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), 'exec'), namespace)
    return namespace[name]


class QCTests(unittest.TestCase):
    def test_geometry_detects_shift_direction_spacing_and_size(self):
        ref = sitk.Image([5, 5, 5], sitk.sitkUInt8)
        self.assertEqual(geometry_metrics(ref, ref)['geometry_status'], 'matched')
        for field, value in [('Origin', (1, 0, 0)), ('Spacing', (2, 1, 1)),
                             ('Direction', (-1, 0, 0, 0, 1, 0, 0, 0, 1))]:
            changed = sitk.Image(ref)
            getattr(changed, 'Set' + field)(value)
            self.assertEqual(geometry_metrics(ref, changed)['geometry_status'], 'mismatch')
        self.assertFalse(geometry_metrics(ref, sitk.Image([4, 5, 5], sitk.sitkUInt8))['size_match'])

    def test_checkpoint_and_no_automatic_visual_acceptance(self):
        with tempfile.TemporaryDirectory() as tmp:
            q = QCReport(tmp)
            q.start_case('test')
            q.organ('test', 'PET', 'liver', 'extracted')
            self.assertEqual(q.summary()[0]['technical_status'], 'incomplete')
            q.stage('test', 'visualization', 'completed')
            q.stage('test', 'case', 'finished')
            self.assertEqual(q.summary()[0]['technical_status'], 'checks_completed')
            self.assertEqual(q.summary()[0]['visual_review'], 'pending')
            q.stage('test', 'visualization', 'failed')
            self.assertEqual(q.summary()[0]['technical_status'], 'attention_required')
            with (q.directory / 'case_qc.csv').open() as f:
                self.assertEqual(list(csv.DictReader(f))[0]['technical_status'], 'attention_required')
            self.assertNotEqual(QCReport(tmp).directory, q.directory)

    def test_nonfinite_values(self):
        self.assertEqual(nonfinite_features({'original_x': np.nan, 'original_y': np.inf,
                                            'original_z': 1, 'diagnostics_x': 'text'}),
                         ['original_x', 'original_y'])

    def test_extraction_records_missing_empty_small_geometry_and_nonfinite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ref = sitk.Image([5, 5, 5], sitk.sitkFloat32)
            ct = root / 'ct.nii.gz'
            sitk.WriteImage(ref, str(ct))
            organs = ['missing', 'empty', 'small', 'shifted', 'good', 'nonfinite', 'error']
            for organ in organs[1:]:
                mask = sitk.Image([5, 5, 5], sitk.sitkUInt8)
                if organ != 'empty':
                    mask += 1
                if organ == 'small':
                    mask *= 0
                    mask[0, 0, 0] = 1
                if organ == 'shifted':
                    mask.SetOrigin((3, 0, 0))
                sitk.WriteImage(mask, str(root / (organ + '.nii.gz')))
            extractor = Mock()
            extractor.execute.side_effect = [dict(original_firstorder_Mean=2.0),
                                             dict(original_firstorder_Mean=np.nan), ValueError('synthetic')]
            shape = Mock()
            shape.execute.return_value = dict(original_shape_VoxelVolume=125)
            env = dict(Path=Path, sitk=sitk, np=np, SEG_DIR=root,
                       CONFIG={'organs': organs, 'output': {}},
                       get_minimum_roi_size=lambda *a: 50,
                       create_pet_grid_extractors=lambda *a: (extractor, shape),
                       log_progress=lambda *a: None, geometry_metrics=geometry_metrics,
                       nonfinite_features=nonfinite_features,
                       merge_pet_and_shape_results=merge_pet_and_shape_results)
            load_function('resample_mask_to_image', env)
            extract = load_function('step3_radiomics', env)
            q = QCReport(root)
            q.start_case('synthetic')
            rows = extract(ct, root, 'synthetic', 'PET', shape_image_path=ct, qc=q)
            self.assertEqual(len(rows), 2)
            self.assertEqual([r['status'] for r in q.organs],
                             ['missing_mask', 'empty_mask', 'small_roi', 'source_geometry_mismatch',
                              'extracted', 'nonfinite_features', 'extraction_failed'])
            self.assertEqual(q.organs[4]['geometry_status'], 'matched')
            self.assertEqual(q.organs[5]['nonfinite_count'], 1)
            env['CONFIG']['organs'] = ['good', 'nonfinite']
            def empty_transfer(source, ref_image, dest):
                blank = sitk.Image(ref_image.GetSize(), sitk.sitkUInt8)
                blank.CopyInformation(ref_image)
                sitk.WriteImage(blank, str(dest))
                return True
            env['resample_mask_to_image'] = Mock(side_effect=[False, True])
            # Existing transferred nonfinite mask is replaced with an empty mask.
            empty_transfer(None, ref, root / 'synthetic_PET_native_grid' / 'nonfinite.nii.gz')
            extract(ct, root, 'synthetic', 'PET', shape_image_path=ct, qc=q)
            self.assertEqual([r['status'] for r in q.organs[-2:]],
                             ['resampling_failed', 'empty_after_transfer'])
            env['CONFIG']['organs'] = ['all']
            env['expected_masks'] = lambda *a: ['missing', 'empty']
            env['resample_mask_to_image'] = empty_transfer
            extract(ct, root, 'synthetic', 'PET', shape_image_path=ct, qc=q)
            self.assertEqual([r['status'] for r in q.organs[-2:]], ['missing_mask', 'empty_mask'])

    def test_failed_run_writes_qc_and_removes_stale_case_rows_with_backup(self):
        import sys
        import os
        tree = ast.parse(SOURCE.read_text())
        entry = next(n for n in tree.body if isinstance(n, ast.If)
                     and isinstance(n.test, ast.Compare) and isinstance(n.test.left, ast.Name)
                     and n.test.left.id == '__name__')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = root / 'input'
            (inputs / 'synthetic').mkdir(parents=True)
            pd.DataFrame([dict(PatientID='current', Modality='PET', Organ='liver'),
                          dict(PatientID='other', Modality='PET', Organ='liver')]).to_csv(root / 'results.csv', index=False)
            config = dict(output={'csv_file': 'results.csv'}, modalities=['CT', 'PET'], organs=['liver'])
            env = dict(__name__='__main__', Path=Path, ROOT_DIR=root, DICOM_DIR=inputs,
                       NIFTI_DIR=root, SEG_DIR=root, QCReport=QCReport, pd=pd, os=os, sys=sys,
                       load_config=lambda *a: config, log_progress=lambda *a: None,
                       check_gpu_memory=lambda: (False, 'synthetic'),
                       get_or_create_anon_id=lambda *a: 'current',
                       detect_folder_structure=lambda *a: {'CT': inputs},
                       step1_convert_dicom=lambda *a: None,
                       should_extract_ct_radiomics=lambda *a: False)
            with patch.object(sys, 'argv', ['run_pipeline.py']):
                exec(compile(ast.Module(body=[entry], type_ignores=[]), str(SOURCE), 'exec'), env)
            q = env['qc']
            self.assertTrue((q.directory / 'previous_radiomics_results.csv').exists())
            self.assertTrue(pd.read_csv(q.directory / 'radiomics_results.csv').empty)
            self.assertEqual(list(pd.read_csv(root / 'results.csv').PatientID), ['other'])
            self.assertEqual(q.organs[0]['status'], 'not_attempted')
            self.assertEqual(q.summary()[0]['technical_status'], 'attention_required')
            self.assertTrue(any(r['stage'] == 'visualization' and r['status'] == 'failed' for r in q.stages))

            # A fresh successful extraction with image generation failing must
            # never reuse a historical PNG or declare technical completion.
            img_path = root / 'image.nii.gz'
            sitk.WriteImage(sitk.Image([5, 5, 5], sitk.sitkFloat32), str(img_path))
            def display_copy(pet, ct, dest):
                import shutil
                shutil.copy2(ct, dest)
                return True
            def extract(*args, qc, **kwargs):
                qc.organ('current', 'PET', 'liver', 'extracted')
                return [dict(PatientID='current', Modality='PET', Organ='liver', original_firstorder_Mean=2)]
            env.update(detect_folder_structure=lambda *a: {'CT': inputs, 'PET': inputs},
                       step1_convert_dicom=lambda *a: img_path,
                       step2_segmentation=lambda *a, **k: root,
                       convert_pet_to_suv=lambda *a: True,
                       resample_pet_to_ct_space=display_copy,
                       step3_radiomics=extract, geometry_metrics=geometry_metrics, sitk=sitk)
            with patch.object(sys, 'argv', ['run_pipeline.py']), patch(
                    'visualize_mask_verification.create_mask_verification', side_effect=RuntimeError('synthetic')):
                exec(compile(ast.Module(body=[entry], type_ignores=[]), str(SOURCE), 'exec'), env)
            q = env['qc']
            self.assertEqual(q.summary()[0]['extracted_rows'], 1)
            self.assertEqual(q.summary()[0]['technical_status'], 'attention_required')
            self.assertTrue(any(r['stage'] == 'visualization' and r['status'] == 'failed' for r in q.stages))
            self.assertEqual(set(pd.read_csv(root / 'results.csv').PatientID), {'current', 'other'})
            aggregate = pd.read_csv(root / 'results.csv')
            current = aggregate[aggregate.PatientID == 'current'].iloc[0]
            self.assertEqual(current.RunID, q.run_id)
            self.assertEqual(len(current.PipelineSHA256), 64)
            self.assertTrue(pd.isna(aggregate[aggregate.PatientID == 'other'].iloc[0].RunID))
            png = root / 'synthetic.png'
            png.touch()
            with patch.object(sys, 'argv', ['run_pipeline.py']), patch(
                    'visualize_mask_verification.create_mask_verification', return_value=png):
                exec(compile(ast.Module(body=[entry], type_ignores=[]), str(SOURCE), 'exec'), env)
            q = env['qc']
            self.assertEqual(q.summary()[0]['technical_status'], 'checks_completed')
            self.assertEqual(q.summary()[0]['visual_review'], 'pending')
            self.assertTrue(any(r['artifact'] == str(png) for r in q.stages))


if __name__ == '__main__':
    unittest.main()
