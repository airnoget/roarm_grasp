"""Check printed AprilTags and 7x5 chessboard corners; no arm connection."""
import cv2
import numpy as np
from camera import orbbec


def annotate(image):
    view=image.copy()
    detector=cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11))
    corners,ids,_=detector.detectMarkers(image)
    if ids is not None: cv2.aruco.drawDetectedMarkers(view,corners,ids)
    found,chess=cv2.findChessboardCornersSB(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),(7,5))
    if found: cv2.drawChessboardCorners(view,(7,5),chess,True)
    tags=[] if ids is None else ids.flatten().tolist()
    status=f'tag36h11 IDs: {tags} | chess 7x5: {found}'
    cv2.putText(view,status,(8,24),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,255,0),2)
    return view,status


def main():
    print('Board recognition only. No arm port is opened. Q/Esc exits.',flush=True)
    try:
        with orbbec(with_color=True) as read:
            for _ in range(15): read()
            while True:
                frame=read()
                aligned,status=annotate(frame.bgr)
                valid=np.isfinite(frame.depth_m)&(frame.depth_m>0)&(frame.depth_m<10)
                cv2.putText(aligned,f'Depth valid: {100*valid.mean():.1f}% - not calibrated',
                            (8,48),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,255),1)
                cv2.namedWindow('Calibration board - depth-aligned RGB',cv2.WINDOW_NORMAL)
                cv2.imshow('Calibration board - depth-aligned RGB',aligned)
                if frame.raw_bgr is not None:
                    original,_=annotate(frame.raw_bgr)
                    cv2.namedWindow('Calibration board - original RGB',cv2.WINDOW_NORMAL)
                    cv2.imshow('Calibration board - original RGB',original)
                if cv2.waitKey(1)&255 in (ord('q'),27): break
    finally: cv2.destroyAllWindows()


if __name__=='__main__': main()
