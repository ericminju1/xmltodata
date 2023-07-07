import pretty_midi
import copy
import mido

def length(midi_path):
    mid = pretty_midi.PrettyMIDI(midi_path)
    mid2 = copy.deepcopy(mid)
    length = mid2.get_end_time()
    return length

def split_midi(base, split_num, start_time, end_time, export=True):
    mid = pretty_midi.PrettyMIDI(base)
    mid2 = copy.deepcopy(mid)
    length = mid2.get_end_time()
    if start_time > length:
        raise Exception("start time exceeds length")
    if end_time > length:
        end_time = length
    
    mid2.adjust_times([start_time, end_time], [0,length])
    mid2.adjust_times([0,length], [0,end_time-start_time])
    
    num_inst = len(mid2.instruments)
    
    for N in range(num_inst):
        notes = mid2.instruments[N].notes
        del_idx = []
        for idx, note in enumerate(notes):
            if note.pitch < 10:
                del_idx.append(idx)
        for index in sorted(del_idx, reverse=True):
            del notes[index]
            
    for N in range(num_inst):
        notes = mid2.instruments[N].notes
        del_idx = []
        end_max = 0
        for idx, note in enumerate(notes):
            if note.start < end_max - (note.end - note.start) * 0.2:
                del_idx.append(idx)
            if note.end > end_max:
                end_max = note.end
        for index in sorted(del_idx, reverse=True):
            del notes[index] 
            
    if export:
        mid2.write("{}/{:02d}.mid".format(base[:-4], split_num))
    
    return mid2