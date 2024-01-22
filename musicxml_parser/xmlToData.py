from .scoreToDurationList import scoreToPianorolld
from .scoreToPianoroll import scoreToPianoroll
import numpy as np

def xmlToData(score_path, quantization=96):
    pianoroll, articulation, score = scoreToPianorolld(score_path, quantization)
    key = list(pianoroll.keys())[0]

    shortest_notes = {}
    for d_key in score.duration_list.keys():
        lst = score.duration_list[d_key]
        if len(lst)//20 > 1:
            shortest_notes[d_key] = np.mean(sorted(lst)[:len(lst)//20])

    pianoroll, articulation, score = scoreToPianoroll(score_path, quantization, shortest_notes)

    return pianoroll[key], articulation[key], score.dynamics