from xmlToData import xmlToData
from reverse_pianoroll import piano_roll_to_pretty_midi

def xmlToMidi(xml_path, quantization=96, program=40, save_path=None):
    roll, art, dyn = xmlToData(xml_path, quantization)
    mid = piano_roll_to_pretty_midi(roll.T, dynamic=dyn, fs=quantization, program=program)
    mid2 = piano_roll_to_pretty_midi(art.T, fs=quantization, program=program)

    if save_path is None:
        save_path = ".".join(xml_path.split('.')[:-1])
        
    mid.write("{}_roll.mid".format(save_path))
    mid2.write("{}_art.mid".format(save_path))