from xmlToData import xmlToData
from reverse_pianoroll import piano_roll_to_pretty_midi

def xmlToMidi(xml_paths, quantization=96, programs=[40,40,40,40,41,41,42,42], save_path=None):
    rolls = []
    # arts = []
    dyns = []

    for xml_path in xml_paths:
        print('---')
        roll, art, dyn = xmlToData(xml_path, quantization)
        rolls.append(roll.T)
        rolls.append(art.T)
        dyns.append(dyn)
        dyns.append(None)

    mid = piano_roll_to_pretty_midi(rolls, dynamics=dyns, fs=quantization, programs=programs)

    if save_path is None:
        save_path = ".".join(xml_paths[0].split('.')[:-1])
        
    mid.write("{}_all.mid".format(save_path))

