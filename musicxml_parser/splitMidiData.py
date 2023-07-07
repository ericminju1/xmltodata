from split_midi import split_midi
import pretty_midi
import os

def splitMidiData(roll_path, art_path):
    mid = pretty_midi.PrettyMIDI(roll_path)
    times, tempos = mid.get_tempo_changes()
    end_time = mid.get_end_time()

    for directory in ([roll_path[:-4], art_path[:-4]]):
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    times = list(times) + [end_time]
    for i in range(len(times)-1):
        mid_split = split_midi(roll_path, i, times[i], times[i+1])
        mid_split2 = split_midi(art_path, i, times[i], times[i+1])