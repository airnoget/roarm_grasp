from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'.deps'))
import unittest
from types import SimpleNamespace as S
import numpy as np
from realsense_geometry import factory_metadata
from camera import Frame,pinhole_rays,save_frame,load_frame
from rgbd_projection import depth_to_rgb,supports_native
import tempfile


def profiles(model='distortion.none'):
    intr=S(fx=100,fy=100,ppx=100,ppy=100,coeffs=[0]*5,model=model)
    color=S(as_video_stream_profile=lambda:S(get_intrinsics=lambda:intr))
    rotation=np.array([[0,-1,0],[1,0,0],[0,0,1]],float)
    depth=S(get_extrinsics_to=lambda _:S(rotation=rotation.flatten(order='F'),translation=[.01,0,0]))
    return depth,color,rotation


class RealSenseGeometryTests(unittest.TestCase):
    def test_column_major_rotation_and_millimetres(self):
        d,c,r=profiles();k,dist,t,model=factory_metadata(d,c)
        np.testing.assert_allclose(t[:3,:3],r)
        np.testing.assert_allclose(t[:3,3],[10,0,0])
        frame=Frame(np.ones((10,20)),pinhole_rays((10,20),100,100,0,0),
                    np.zeros((10,20,3),np.uint8),np.zeros((200,200,3),np.uint8),k,dist,t,model)
        self.assertTrue(supports_native(frame))
        np.testing.assert_allclose(depth_to_rgb(frame)[4,8],[97,108],atol=1e-5)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'rgbd.npz';save_frame(path,frame)
            loaded=load_frame(path)
            np.testing.assert_allclose(depth_to_rgb(loaded)[4,8],[97,108],atol=1e-5)

    def test_unsupported_models_not_silently_misinterpreted(self):
        for model in ['distortion.inverse_brown_conrady','distortion.modified_brown_conrady']:
            d,c,_=profiles(model)
            with self.assertRaises(ValueError):factory_metadata(d,c)

if __name__=='__main__':unittest.main()
