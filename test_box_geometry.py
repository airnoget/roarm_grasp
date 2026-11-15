from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'.deps'))
import unittest
import numpy as np
from camera import Frame,pinhole_rays
from box_geometry import fit_box


class BoxGeometryTests(unittest.TestCase):
    def frame(self):
        depth=np.full((240,320),.5,np.float32)
        mask=np.zeros(depth.shape,bool);mask[90:150,130:190]=True
        depth[mask]=.46
        return Frame(depth,pinhole_rays(depth.shape,600,600,160,120)),mask

    def test_surface_and_opening_in_mm(self):
        f,m=self.frame();g=fit_box(f,m)
        np.testing.assert_allclose(g['camera_surface_xyz_mm'],[-.3833,-.3833,460],atol=.1)
        self.assertAlmostEqual(g['height_above_support_mm'],40,places=2)
        self.assertTrue(43<g['observed_opening_mm']<47)
        self.assertFalse(g['motion_enabled'])

    def test_missing_centre_remains_missing(self):
        f,m=self.frame();f.depth_m[116:124,156:164]=0
        g=fit_box(f,m)
        self.assertTrue(g['center_is_fitted']);self.assertFalse(g['center_has_depth'])
        self.assertEqual(float(f.depth_m[120,160]),0)

    def test_no_box_or_no_support_rejected(self):
        f,m=self.frame();f.depth_m[:]=.5
        with self.assertRaises(ValueError):fit_box(f,m)

    def test_thin_missing_seam_associated_without_inventing_depth(self):
        f,m=self.frame();f.depth_m[90:150,159:161]=0
        g=fit_box(f,m)
        self.assertAlmostEqual(g['height_above_support_mm'],40,places=2)
        self.assertFalse(g['center_has_depth'])
        self.assertEqual(int(np.count_nonzero(f.depth_m[90:150,159:161])),0)

    def test_widely_separated_surfaces_still_rejected(self):
        f,m=self.frame();f.depth_m[90:150,155:165]=0
        with self.assertRaisesRegex(ValueError,'Multiple separated'):fit_box(f,m)
        f,m=self.frame();f.depth_m[~m]=0
        with self.assertRaises(ValueError):fit_box(f,m)

    def test_native_resolution_fragments_join_only_across_small_physical_gap(self):
        f,m=self.frame();f.rays=pinhole_rays(f.depth_m.shape,1000,1000,160,120)
        f.depth_m[90:150,158:162]=0
        g=fit_box(f,m)
        self.assertAlmostEqual(g['height_above_support_mm'],40,places=2)
        self.assertEqual(int(np.count_nonzero(f.depth_m[90:150,158:162])),0)
        f.depth_m[90:150,154:166]=0
        with self.assertRaisesRegex(ValueError,'Multiple separated'):fit_box(f,m)

if __name__=='__main__':unittest.main()
