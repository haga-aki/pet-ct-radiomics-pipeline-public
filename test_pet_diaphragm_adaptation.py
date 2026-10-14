import unittest
import numpy as np
from pet_diaphragm_adaptation import (
    Parameters, crossings, fit_field, fit_field_shape, get_patches, interval_error,
    inverse_si_coordinates, profile_reliability, warp_labels,
)


class AdaptationTests(unittest.TestCase):
    def test_checkerboard_gaps_do_not_attenuate_constant_displacement(self):
        patches = [dict(training=True, shift=-6., correction=-6.,
                        columns=[(x, y, 30, np.ones(19))])
                   for x in (10, 30, 50) for y in (10, 30, 50)]
        field, _, _ = fit_field(patches, (70, 70), np.array([1., 1., 3.]), Parameters())
        np.testing.assert_allclose(field[10:51, 10:51], -6., atol=1e-12)
        self.assertEqual(field[69, 69], 0.)

    def fitting_patches(self, targets):
        return [dict(training=True, shift=value, correction=value, reliability=1.,
                     unambiguous_fraction=1., crossing_iqr_mm=0.,
                     columns=[(10+i*10, 25, 30, np.ones(19))])
                for i, value in enumerate(targets)]

    def test_constant_displacement_preserves_surface_shape_at_fit_sites(self):
        patches = self.fitting_patches([-6.]*6)
        field, _, _, info = fit_field_shape(patches, (90, 60), np.array([1., 1., 3.]), Parameters())
        self.assertEqual(info['baseline_mm'], -6.)
        self.assertEqual(info['maximum_residual_mm'], 0.)
        for patch in patches:
            x, y, *_ = patch['columns'][0]
            self.assertEqual(field[x, y], -6.)

    def test_unreliable_fit_patch_and_heldout_profiles_have_no_effect(self):
        patches = self.fitting_patches([-6.]*6)
        rejected = self.fitting_patches([8.])[0]
        rejected['unambiguous_fraction'] = .5
        heldout = self.fitting_patches([8.])[0]
        heldout['training'] = False
        parameters = Parameters(organ_bias_gate_mm=1.5)
        a = fit_field_shape(patches, (90, 60), np.array([1., 1., 3.]), parameters)
        b = fit_field_shape(patches+[rejected, heldout], (90, 60), np.array([1., 1., 3.]), parameters)
        for x, y in zip(a[:3], b[:3]):
            np.testing.assert_array_equal(x, y)
        self.assertEqual(a[3], b[3])

    def test_residual_and_total_displacement_are_bounded(self):
        p = Parameters(maximum_local_residual_mm=.2, organ_bias_gate_mm=None)
        f, _, _, info = fit_field_shape(self.fitting_patches([-9., -9., -7., -2., 3., 8.]),
                                      (90, 60), np.array([1., 1., 3.]), p)
        self.assertLessEqual(info['maximum_residual_mm'], .2+1e-12)
        self.assertLessEqual(np.max(np.abs(f)), p.max_shift_mm)

    def test_outlier_does_not_replace_representative_displacement(self):
        f, _, _, info = fit_field_shape(self.fitting_patches([-6., -6., -6., -6., -6., 8.]),
                                      (90, 60), np.array([1., 1., 3.]), Parameters())
        self.assertEqual(info['baseline_mm'], -6.)
        self.assertLess(np.max(f), .01)
        self.assertLess(info['maximum_residual_mm'], 2.)

    def test_local_residual_can_represent_broad_spatial_variation(self):
        f, _, _, _ = fit_field_shape(self.fitting_patches([-7., -6.5, -6., -5., -4.5, -4.]),
                                    (90, 60), np.array([1., 1., 3.]), Parameters())
        self.assertGreater(f[60, 25]-f[10, 25], .1)

    def test_ambiguous_crossings_reduce_reliability(self):
        z = (np.arange(-9, 10)-.5)*3
        clean = np.tile(1/(1+np.exp(z/3)), (8, 1))
        good = profile_reliability(clean, 3., .7)
        altered = clean.copy()
        altered[:4, 12:16] = 1.
        bad = profile_reliability(altered, 3., .7)
        self.assertEqual(good['reliability'], 1.)
        self.assertLess(bad['reliability'], good['reliability'])

    def test_median_alone_cannot_activate_inconsistent_targets(self):
        p = Parameters(organ_bias_gate_mm=1.5)
        f, _, _, info = fit_field_shape(self.fitting_patches([-2., 0., 0., 2., 3., 4., 5.]),
                                      (90, 60), np.array([1., 1., 3.]), p)
        self.assertGreater(info['baseline_mm'], 1.5)
        self.assertFalse(info['activated'])
        self.assertFalse(f.any())
        f, _, _, info = fit_field_shape(self.fitting_patches([-6., -6., -5., -4., -3., -2., 0.]),
                                      (90, 60), np.array([1., 1., 3.]), p)
        self.assertTrue(info['activated'])
        self.assertTrue(f.any())

    def test_lung_base_extends_or_retracts_without_apex_translation(self):
        labels = np.zeros((3, 3, 100), np.uint8)
        labels[:, :, 30:90] = 3
        center = np.full((3, 3), 29.5 * 3)
        for displacement, expected_base in ((-6., 28), (6., 32)):
            result = warp_labels(labels, np.full((3, 3), displacement), center, 3., 36.)
            occupied = np.flatnonzero(result[1, 1] == 3)
            self.assertEqual(occupied[0], expected_base)
            self.assertEqual(occupied[-1], 89)
            np.testing.assert_array_equal(result[:, :, 42:], labels[:, :, 42:])

    def test_interface_interval_preserves_anatomical_gap(self):
        np.testing.assert_array_equal(interval_error(np.array([-5., 0., 2., 3., 5.]), 0., 3.),
                                      [-5., 0., 0., 0., 2.])
        with self.assertRaises(ValueError):
            interval_error(1., 3., 0.)
        self.assertTrue(np.isnan(interval_error(np.nan, 0., 3.)))

    def test_inverse_matches_forward_positive_and_negative(self):
        z = np.linspace(-50, 50, 10001)
        for shift in (-9., -3., 0., 3., 9.):
            forward = z + shift * np.maximum(1 - np.abs(z) / 36., 0)
            self.assertTrue(np.all(np.diff(forward) > 0))
            np.testing.assert_allclose(inverse_si_coordinates(forward, 0., shift, 36.), z, atol=1e-12)

    def test_exact_voxel_shift_and_unchanged_remote_regions(self):
        labels = np.zeros((3, 3, 60), np.uint8)
        labels[:, :, 5:30] = 1
        labels[:, :, 30:55] = 3
        shift = np.zeros((3, 3))
        shift[1, 1] = -6
        center = np.full((3, 3), 29.5 * 3)
        result = warp_labels(labels, shift, center, 3., 36.)
        self.assertEqual(np.flatnonzero(result[1, 1] == 1)[-1], 27)
        self.assertEqual(np.flatnonzero(result[1, 1] == 3)[0], 28)
        np.testing.assert_array_equal(result[0], labels[0])
        np.testing.assert_array_equal(result[:, :, :15], labels[:, :, :15])
        np.testing.assert_array_equal(result[:, :, 45:], labels[:, :, 45:])

    def test_zero_field_identity(self):
        labels = np.arange(125, dtype=np.uint8).reshape(5, 5, 5) % 7
        result = warp_labels(labels, np.zeros((5, 5)), np.ones((5, 5)), 3., 36.)
        np.testing.assert_array_equal(result, labels)

    def test_only_si_coordinates_change(self):
        # Distinct labels encode the original x/y column. No destination may
        # acquire a value from a different column, even under a varying field.
        labels = np.zeros((6, 6, 60), np.uint8)
        for x in range(6):
            for y in range(6):
                labels[x, y, 10:50] = 1 + x*6 + y
        shift = np.linspace(-9, 9, 36).reshape(6, 6)
        result = warp_labels(labels, shift, np.full((6, 6), 90.), 3., 36.)
        for x in range(6):
            for y in range(6):
                self.assertTrue(set(np.unique(result[x, y])).issubset({0, 1+x*6+y}))

    def test_heldout_values_do_not_enter_field(self):
        columns = [(x, y, 20, np.ones(19)) for x in range(5, 10) for y in range(5, 10)]
        training = [dict(training=True, shift=-6., columns=columns)]*5
        heldout = dict(training=False, shift=4., columns=columns)
        first = fit_field(training+[heldout], (30, 30), np.array([1, 1, 3]), Parameters())
        self.assertTrue(first[0].any())
        heldout["shift"] = -1000
        second = fit_field(training+[heldout], (30, 30), np.array([1, 1, 3]), Parameters())
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a, b)

    def test_extreme_training_patch_rejected(self):
        patch = dict(training=True, shift=20., columns=[(5, 5, 20, np.ones(19))])
        field = fit_field([patch], (30, 30), np.array([1, 1, 3]), Parameters())
        self.assertFalse(np.any(field[0]))

    def test_organ_bias_gate_uses_training_only(self):
        columns = [(5, 5, 20, np.ones(19))]
        training = [dict(training=True, shift=v, correction=v, columns=columns) for v in [-3., 0., 0., 0., 3.]]
        heldout = [dict(training=False, shift=-8., correction=-8., columns=columns)]*20
        parameters = Parameters(organ_bias_gate_mm=1.5)
        field = fit_field(training+heldout, (30, 30), np.array([1., 1., 3.]), parameters)
        self.assertFalse(np.any(field[0]))
        shifted = [dict(training=True, shift=-6., correction=-6., columns=columns)]*5
        field = fit_field(shifted, (30, 30), np.array([1., 1., 3.]), parameters)
        self.assertTrue(np.any(field[0]))

    def test_known_profile_crossing(self):
        z = (np.arange(-9, 10) - .5) * 3
        for target in (-6, 0, 6):
            profile = 1 / (1 + np.exp((z - target) / 3))
            measured = crossings(profile, 3., .7)[.5]
            self.assertAlmostEqual(measured, target, places=5)

    def test_known_synthetic_dome_error_reduced_in_heldout_columns(self):
        shape = (80, 80, 90)
        x, y = np.indices(shape[:2])
        radius = ((x - 40)**2 + (y - 40)**2) / 32**2
        footprint = radius < 1
        true_top = np.floor(49 - 6*radius).astype(int)
        # Deliberately introduce a +6-mm cranial CT-mask error.
        original_top = true_top + 2
        z = np.arange(shape[2])[None, None, :]
        organ = footprint[:, :, None] & (z >= 15) & (z <= original_top[:, :, None])
        lung = footprint[:, :, None] & (z >= original_top[:, :, None]+2) & (z < 80)
        distance = (z - true_top[:, :, None] - .5)*3
        pet = .25 + 2.25/(1 + np.exp(distance/3))
        parameters = Parameters()
        patches = get_patches(pet, organ, lung, np.array([1., 1., 3.]), parameters)
        shift, center, _ = fit_field(patches, shape[:2], np.array([1., 1., 3.]), parameters)
        labels = organ.astype(np.uint8) + 3*lung.astype(np.uint8)
        adapted = warp_labels(labels, shift, center, 3., parameters.band_mm)
        errors = []
        for patch in patches:
            if patch["training"]:
                continue
            for cx, cy, _, _ in patch["columns"]:
                if radius[cx, cy] < .5:
                    top = np.flatnonzero(adapted[cx, cy] == 1)[-1]
                    errors.append(abs(top - true_top[cx, cy])*3)
        self.assertGreater(len(errors), 100)
        self.assertLessEqual(np.median(errors), 1.5)


if __name__ == "__main__":
    unittest.main()
