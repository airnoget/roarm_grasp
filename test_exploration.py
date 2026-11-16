import unittest
from exploration import TargetRegistry,CoverageGrid,ExplorationCoordinator,PreviewTransport,ScanEnvelope
from target_selection import selected_proposal


class ExplorationTests(unittest.TestCase):
    def test_reordered_detections_keep_id_and_lock_nearest(self):
        r=TargetRegistry();a=dict(name='cube',position_m=[1,0,0]);b=dict(name='cube',position_m=[.4,0,0])
        self.assertEqual(r.update([a,b],1),[1,2]);self.assertEqual(r.choose([0,0,0],1),2)
        self.assertEqual(r.update([b,a],2),[2,1]);self.assertEqual(r.choose([1,0,0],2),2)
        r.update([],3);c=ExplorationCoordinator(r,PreviewTransport())
        self.assertEqual(c.tick([0,0,0],3)['intent']['mode'],'stop')
        self.assertEqual(r.update([b],4),[2]);r.collected();self.assertEqual(r.tracks[2]['state'],'collected')

    def test_ambiguous_same_colour_not_merged(self):
        r=TargetRegistry();r.update([dict(name='cube',position_m=[0,0,0])],1)
        self.assertEqual(r.update([dict(name='cube',position_m=[.01,0,0]),dict(name='cube',position_m=[-.01,0,0])],2),[None,None])

    def test_walls_unknown_and_clearance(self):
        g=CoverageGrid([[0,1,0,-1],[0,1,0,-1],[0,0,0,-1]],1)
        g.observe((0,0),0,1.5,5)
        self.assertNotIn((2,0),g.seen);self.assertNotIn((3,0),g.seen)
        plan=g.next_scan((0,0));self.assertEqual(plan['unknown'],3)
        self.assertTrue(all(g.free(c) for c in plan['path']))
        self.assertFalse(CoverageGrid([[0]*3 for _ in range(3)],1,clearance_m=1).free((0,0)))

    def test_complete_floor_does_not_mean_unknown_complete(self):
        g=CoverageGrid([[0,-1]],1);g.seen.add((0,0))
        self.assertEqual(g.next_scan((0,0))['unknown'],1)

    def test_selection_preserves_other_objects_and_rejects_missing(self):
        result=dict(selected_target_id=2,native_detections=[{},{}],proposals=[dict(target_id=1),dict(target_id=2)])
        self.assertEqual(selected_proposal(result)['target_id'],2);self.assertEqual(len(result['proposals']),2)
        result['selected_target_id']=3
        with self.assertRaises(ValueError):selected_proposal(result)

    def test_untaught_scan_cannot_generate_movement(self):
        with self.assertRaises(ValueError):ScanEnvelope(dict(poses={},edges=[])).edge('left','right')


if __name__=='__main__':unittest.main()
