from .xmlToData import xmlToData
from .reverse_pianoroll import piano_roll_to_pretty_midi
import numpy as np

def monophonic(pianoroll):
    roll_nonzero = (pianoroll != 0)
    for i in range(pianoroll.shape[0]):
        if np.sum(roll_nonzero[i]) > 1:
            pitches = [i for i, x in enumerate(roll_nonzero[i]) if x]
            highest_pitch = np.max(pitches)
            pianoroll[i][:highest_pitch] = 0
    return pianoroll

def xmlToMidi(xml_paths, quantization=96, programs=[40,40,40,40,41,41,42,42], save_path=None):
    rolls = []
    # arts = []
    dyns = []

    for xml_path in xml_paths:
        print('---')
        roll, art, dyn = xmlToData(xml_path, quantization)

        # only for strictly monophonic model
        roll = monophonic(roll)
        art = monophonic(art)

        rolls.append(roll.T)
        rolls.append(art.T)
        dyns.append(dyn)
        dyns.append(None)

    mid = piano_roll_to_pretty_midi(rolls, dynamics=dyns, fs=quantization, programs=programs)

    if save_path is None:
        save_path = ".".join(xml_paths[0].split('.')[:-1])
        
    mid.write("{}_all.mid".format(save_path))