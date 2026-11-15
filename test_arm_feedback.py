from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'.deps'))
import json
import unittest
from arm import Arm

class Port:
    def __init__(self,lines):self.lines=list(lines);self.writes=[]
    def reset_input_buffer(self):pass
    def write(self,value):self.writes.append(json.loads(value))
    def readline(self):return self.lines.pop(0) if self.lines else b''

class Tests(unittest.TestCase):
    def arm(self,lines):
        arm=Arm.__new__(Arm);arm.port=Port(lines);return arm
    def test_fragmented_five_joint_feedback_without_gripper(self):
        message=json.dumps(dict(T=1051,b=0,s=0,e=1,t=0,r=.8,v=1200)).encode()+b'\n'
        arm=self.arm([b'booting\n',message[:15],b'',message[15:]])
        self.assertEqual(arm.feedback(required_fields=('b','s','e','t','r','v'))['v'],1200)
        self.assertTrue(all(m=={'T':105} for m in arm.port.writes))
    def test_grip_execution_default_still_rejects_missing_fields(self):
        arm=self.arm([b'{"T":1051,"x":0,"y":0,"z":0,"tit":0,"r":0}\n'])
        with self.assertRaisesRegex(TimeoutError,'missing_or_invalid'):arm.feedback(timeout=.01)
    def test_nonfinite_joint_not_accepted(self):
        arm=self.arm([b'{"T":1051,"b":NaN}\n'])
        with self.assertRaisesRegex(TimeoutError,'b'):arm.feedback(timeout=.01,required_fields=('b',))

if __name__=='__main__':unittest.main()
