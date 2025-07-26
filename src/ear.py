
import numpy as np
LEFT_EYE = [33,160,158,133,153,144]
RIGHT_EYE = [263,385,387,362,373,380]

def _ear(eye):
    A = np.linalg.norm(eye[1]-eye[5]) + np.linalg.norm(eye[2]-eye[4])
    B = 2.0*np.linalg.norm(eye[0]-eye[3])
    return A / B

def average_ear(landmarks):
    """Compute mean EAR for both eyes given 468x3 face-mesh array."""
    eye_l = np.array([landmarks[i][:2] for i in LEFT_EYE])
    eye_r = np.array([landmarks[i][:2] for i in RIGHT_EYE])
    return (_ear(eye_l)+_ear(eye_r))/2.0
