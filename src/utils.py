
import pathlib
import numpy as np

# Eye and mouth landmark indices you need for EAR and mouth crop
LEFT_EYE  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [263, 387, 385, 362, 380, 373]
MOUTH_OUTER = [78, 308, 13, 14, 87, 317, 82, 312]

def play_alert():
    import simpleaudio as sa
    import sys
    wav = pathlib.Path(__file__).resolve().parent.parent / 'assets' / 'alarm.wav'
    try:
        wave_obj = sa.WaveObject.from_wave_file(str(wav))
        wave_obj.play()  # Non-blocking
    except Exception as e:
        print('[WARN] Cannot play sound:', e)


