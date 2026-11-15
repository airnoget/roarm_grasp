import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'.deps'))
import numpy as np
from rear_transfer import RearTransfer,half_turn_target

class RearTransferTests(unittest.TestCase):
    def test_half_turn_at_limits_preserves_valid_encoder_target(self):
        for degrees in (-179.9,-90,0,.44,90,179.9):
            target=half_turn_target(np.radians(degrees))
            self.assertAlmostEqual(abs(np.degrees(target)-degrees),180,delta=.1)
            self.assertTrue(0<=round(2047-target*4096/(2*np.pi))<=4095)

    def test_only_base_moves_then_jaw_releases_after_arrival(self):
        class Arm:
            def __init__(self):self.commands=[]
            def send(self,command):self.commands.append(command)
        arm=Arm();transfer=RearTransfer(arm)
        state=dict(b=np.radians(.44),g=np.radians(135.1),v=1235)
        transfer.start(state)
        for _ in range(5):transfer.update(state)
        self.assertEqual([c['joint'] for c in arm.commands],[1])
        state['b']=transfer.target
        for _ in range(3):transfer.update(state)
        self.assertEqual([c['joint'] for c in arm.commands],[1,1,6])
        self.assertEqual([c['spd'] for c in arm.commands],[320,80,120])
        self.assertEqual(arm.commands[0]['rad'],arm.commands[1]['rad'])
        self.assertAlmostEqual(np.degrees(arm.commands[-1]['rad']),110)
        state['g']=np.radians(110)
        for _ in range(3):transfer.update(state)
        self.assertIsNone(transfer.phase)

    def test_stop_does_not_command_shoulder_elbow_or_wrists(self):
        class Arm:
            def __init__(self):self.commands=[]
            def send(self,command):self.commands.append(command)
        arm=Arm();transfer=RearTransfer(arm);state=dict(b=0,g=2.,v=1235)
        transfer.start(state);transfer.stop(state)
        self.assertEqual([c['joint'] for c in arm.commands],[1,1,6])

if __name__=='__main__':unittest.main()
