# https://github.com/craffel/pretty-midi/blob/main/examples/reverse_pianoroll.py

"""
Utility function for converting an audio file
to a pretty_midi.PrettyMIDI object. Note that this method is nowhere close
to the state-of-the-art in automatic music transcription.
This just serves as a fun example for rough
transcription which can be expanded on for anyone motivated.
"""
from __future__ import division
import sys
import argparse
import numpy as np
import pretty_midi
import librosa


def piano_roll_to_pretty_midi(piano_rolls, dynamics=None, fs=100, programs=None):
    '''Convert a Piano Roll array into a PrettyMidi object
     with a single instrument.

    Parameters
    ----------
    piano_roll : np.ndarray, shape=(128,frames), dtype=int
        Piano roll of one instrument
    fs : int
        Sampling frequency of the columns, i.e. each column is spaced apart
        by ``1./fs`` seconds.
    program : int
        The program number of the instrument.

    Returns
    -------
    midi_object : pretty_midi.PrettyMIDI
        A pretty_midi.PrettyMIDI class instance describing
        the piano roll.

    '''
    if programs is None:
        l = len(piano_rolls)
        program = [0]*l
    
    if dynamics is None:
        l = len(piano_rolls)
        dynamics = [None]*l

    pm = pretty_midi.PrettyMIDI()

    for piano_roll, dynamic, program in zip(piano_rolls, dynamics, programs):
        notes, frames = piano_roll.shape
        instrument = pretty_midi.Instrument(program=program)

        # pad 1 column of zeros so we can acknowledge inital and ending events
        piano_roll = np.pad(piano_roll, [(0, 0), (1, 1)], 'constant')

        # use changes in velocities to find note on / note off events
        velocity_changes = np.nonzero(np.diff(piano_roll).T)

        if dynamic is not None:
            dynamic = (dynamic*127.9).astype(np.int32)
            dynamic_changes = np.nonzero(np.diff(dynamic))

        # keep track on velocities and note on times
        prev_velocities = np.zeros(notes, dtype=int)
        note_on_time = np.zeros(notes)

        for time, note in zip(*velocity_changes):
            # use time + 1 because of padding above
            velocity = piano_roll[note, time + 1]
            time = time / fs
            if velocity > 0:
                if prev_velocities[note] == 0:
                    note_on_time[note] = time
                    prev_velocities[note] = velocity
            else:
                pm_note = pretty_midi.Note(
                    velocity=80,
                    pitch=note+12,
                    start=note_on_time[note],
                    end=time)
                instrument.notes.append(pm_note)
                prev_velocities[note] = 0
        
        if dynamic is not None:
            for time, in zip(*dynamic_changes):
                dyn = dynamic[time + 1]
                pm_dynamic = pretty_midi.ControlChange(
                    number=11,
                    value=dyn,
                    time=(time+1) / fs
                )
                instrument.control_changes.append(pm_dynamic)

        pm.instruments.append(instrument)

    return pm
