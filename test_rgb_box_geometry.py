from pathlib import Path
import sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parent/'.deps'))
import numpy as np
from camera import Frame,pinhole_rays
from rgbd_projection import depth_to_rgb
from rgb_box_geometry import fit_rgb_top

class RGBBoundaryTests(unittest.TestCase):
    def scene(self):
        depth=np.full((240,320),.5,np.float32)
        mask=np.zeros(depth.shape,bool);mask[90:150,130:190]=True
        depth[mask]=.46;depth[90:150,154:166]=0
        image=np.full((240,320,3),190,np.uint8);image[90:150,130:190]=[60,130,60]
        k=np.array([[600.,0,160],[0,600,120],[0,0,1]])
        frame=Frame(depth,pinhole_rays(depth.shape,600,600,160,120),image.copy(),image,k,np.zeros(8),np.eye(4),'OBDistortionModel.NONE')
        return frame,mask

    def test_rgb_boundary_with_measured_plane_recovers_seam_without_filling(self):
        f,m=self.scene();before=f.depth_m.copy()
        result=fit_rgb_top(f,m,m,[125,85,195,155],depth_to_rgb(f))
        np.testing.assert_allclose(result['camera_surface_xyz_mm'],[-.383,-.383,460],atol=2)
        self.assertTrue(42<result['observed_opening_mm']<48)
        np.testing.assert_array_equal(f.depth_m,before)
        self.assertFalse(result['center_has_depth'])
        self.assertIn('RGB',result['boundary_source'])

    def test_no_depth_cannot_be_replaced_by_rgb_outline(self):
        f,m=self.scene();f.depth_m[m]=0
        with self.assertRaises(ValueError):fit_rgb_top(f,m,m,[125,85,195,155],depth_to_rgb(f))

    def test_no_visual_boundary_is_rejected(self):
        f,m=self.scene();f.raw_bgr[:]=190
        with self.assertRaisesRegex(ValueError,'no depth-supported'):fit_rgb_top(f,m,m,[125,85,195,155],depth_to_rgb(f))

if __name__=='__main__':unittest.main()
