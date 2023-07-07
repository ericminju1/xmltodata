from split_midi import split_midi
import pretty_midi
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
        mid_split = split_midi(path, i, times[i], times[i+1], export=False)
        pm = pretty_midi.PrettyMIDI()
        pm2 = pretty_midi.PrettyMIDI()
        for n, inst in enumerate(mid_split.instruments):
            if n%2 == 0:
                pm.instruments.append(inst)
            else:
                pm2.instruments.append(inst)
        pm.write("{}/{:02d}_roll.mid".format(path[:-4], i))
        pm2.write("{}/{:02d}_art.mid".format(path[:-4], i))