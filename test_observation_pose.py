from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'.deps'))
import unittest
import numpy as np
from observation_pose import compare


class ObservationPoseTests(unittest.TestCase):
    def test_screenshot_reference_is_joint_degrees(self):
        raw=np.radians([.44,7.47,130.34,-2.20,47.81])
        result=compare(raw)
        self.assertLess(result['max_difference_deg'],1e-10)
        self.assertFalse(result['motion_enabled'])

    def test_actual_measurement_is_not_replaced_by_reference(self):
        raw=np.radians([.44,7.47,116.34,-2.20,47.81])
        result=compare(raw)
        self.assertAlmostEqual(result['measured_joint_deg'][2],116.34)
        self.assertAlmostEqual(result['difference_deg'][2],-14)

if __name__=='__main__':unittest.main()
