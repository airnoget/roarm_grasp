import unittest
from geometry_tracking import GeometryTracker


def proposal(x=0,width=50,axis=(1,0,0)):
    return dict(camera_xyz_mm=[x,0,300],geometry=dict(camera_closing_axis=axis,observed_opening_mm=width))


class TrackingTests(unittest.TestCase):
    def test_five_consistent_samples_and_sign_symmetry(self):
        t=GeometryTracker()
        for i in range(5):
            p=proposal(x=(i%2)*.5,axis=((-1)**i,0,0));t.update([p],i*.1)
            self.assertEqual(p['tracking']['camera_relative_stable'],i==4)
            self.assertFalse(p['tracking']['motion_authorized'])

    def test_order_change_keeps_identity(self):
        t=GeometryTracker();a,b=proposal(),proposal(100);t.update([a,b],0)
        c,d=proposal(100),proposal();t.update([c,d],.1)
        self.assertEqual(a['tracking']['id'],d['tracking']['id'])
        self.assertEqual(b['tracking']['id'],c['tracking']['id'])

    def test_gap_missing_or_invalid_resets(self):
        for reset in ('gap','missing','invalid'):
            t=GeometryTracker()
            for i in range(5):t.update([proposal()],i*.1)
            if reset=='missing':t.update([],.5)
            if reset=='invalid':t.update([proposal()],.5,invalid=True)
            p=proposal();t.update([p],3 if reset=='gap' else .6)
            self.assertFalse(p['tracking']['camera_relative_stable'])
            self.assertEqual(p['tracking']['frames'],1)

    def test_width_drift_and_ambiguous_matches_not_stable(self):
        t=GeometryTracker()
        for i in range(5):
            p=proposal(width=50+i*2);t.update([p],i*.1)
        self.assertFalse(p['tracking']['camera_relative_stable'])
        t=GeometryTracker();t.update([proposal(),proposal(5)],0)
        p=proposal(2);t.update([p],.1)
        self.assertEqual(p['tracking']['frames'],1)

if __name__=='__main__':unittest.main()
