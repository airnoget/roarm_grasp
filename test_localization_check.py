from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'.deps'))
import unittest,tempfile,os
from check_localization import summarize,recording_paths


class LocalizationChecks(unittest.TestCase):
    def test_old_session_is_never_silently_used(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name,stamp in [('preset-old',100),('preset-new',200)]:
                p=root/name/'capture'/'result.json';p.parent.mkdir(parents=True);p.write_text('{}');os.utime(p,(stamp,stamp))
            self.assertEqual(len(recording_paths(root)),1)
            self.assertEqual(recording_paths(root)[0].parent.parent.name,'preset-new')
            self.assertEqual(len(recording_paths(root,all_sessions=True)),2)

    def test_identical_bad_positions_are_not_absolute_accuracy(self):
        rows=[dict(base_surface_mm=[100,0,20],joints_rad=[i*.1,0,0,0,0]) for i in range(3)]
        result=summarize(rows,[90,0,20])
        self.assertEqual(result['max_pair_distance_mm'],0)
        self.assertEqual(result['max_reference_error_mm'],10)
        self.assertFalse(result['absolute_accuracy_verified'])
        self.assertTrue(result['independent_pose_check_available'])

    def test_repeated_pose_not_independent(self):
        result=summarize([dict(base_surface_mm=[100,0,20],joints_rad=[0]*5)]*3)
        self.assertFalse(result['independent_pose_check_available'])

    def test_spread_is_full_pairwise_distance(self):
        rows=[dict(base_surface_mm=[x,0,20],joints_rad=[i*.1,0,0,0,0]) for i,x in enumerate([0,3,6])]
        self.assertEqual(summarize(rows)['max_pair_distance_mm'],6)

if __name__=='__main__':unittest.main()
