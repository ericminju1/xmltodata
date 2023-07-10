from .split_midi import split_midi
import pretty_midi
import mido
import os

def splitMidiData(path):
    mid = pretty_midi.PrettyMIDI(path)
    times, tempos = mid.get_tempo_changes()
    end_time = mid.get_end_time()

    directory = path[:-4]
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    times = list(times) + [end_time]
    for i in range(len(times)-1):
        mid_split, save_path = split_midi(path, i, times[i], times[i+1], export=True)
        midi_data = mido.MidiFile(filename=save_path) 

        for n, inst in enumerate(mid_split.instruments):
            pm = pretty_midi.PrettyMIDI()
            pm._load_tempo_changes(midi_data)
            pm.instruments.append(inst)
            
            if n%2 == 0:
                pm.write("{}/{:02d}_{:01d}_roll.mid".format(path[:-4], i, n//2))
            else:
                pm.write("{}/{:02d}_{:01d}_art.mid".format(path[:-4], i, n//2))