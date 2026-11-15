from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'.deps'))
import unittest
from distance_capture_plan import DistancePlan,banner

class Tests(unittest.TestCase):
    def test_200_with_70_percent_far_and_each_object_both_distances(self):
        p=DistancePlan()
        self.assertEqual(sum(s[3] for s in p.stages),200)
        self.assertEqual(sum(s[3] for s in p.stages if s[1].startswith('far_')),140)
        for distance in ('far','near'):
            p.select(ord('f' if distance=='far' else 'n'))
            for i in range(1,6):
                p.select(ord(str(i)));self.assertEqual(p.active[1],f'{distance}_single_{i}')
                for _ in range(p.active[3]):p.saved()
            for key in ('m','b'):
                p.select(ord(key))
                for _ in range(p.active[3]):p.saved()
        self.assertEqual(p.export()['total_saved'],200)
        self.assertIn('200',p.hint())
    def test_resume_does_not_change_phase_or_counts(self):
        p=DistancePlan();p.select(ord('n'));p.select(ord('5'));p.saved()
        q=DistancePlan();q.restore(p.export());self.assertEqual(q.export(),p.export())
        self.assertEqual(banner(q,1280).shape[1],1280)

if __name__=='__main__':unittest.main()
