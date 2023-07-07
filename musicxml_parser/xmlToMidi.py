from xmlToData import xmlToData
from reverse_pianoroll import piano_roll_to_pretty_midi

def xmlToMidi(xml_paths, quantization=96, programs=[40,40,41,42], save_path=None):
    rolls = []
    arts = []
    dyns = []

    for xml_path in xml_paths:
        print('---')
        roll, art, dyn = xmlToData(xml_path, quantization)
        rolls.append(roll.T)
        arts.append(art.T)
        dyns.append(dyn)

    mid = piano_roll_to_pretty_midi(rolls, dynamics=dyns, fs=quantization, programs=programs)
    mid2 = piano_roll_to_pretty_midi(arts, fs=quantization, programs=programs)

    if save_path is None:
        save_path = ".".join(xml_paths[0].split('.')[:-1])
        
    mid.write("{}_roll.mid".format(save_path))
    mid2.write("{}_art.mid".format(save_path))