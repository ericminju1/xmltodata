from midiToAudio import midiToAudio
from midiToSynth import midiToSynth
from synthToAudio import synthToAudio

def generate_dataset(path_format, save_format, splits, insts=4):
    for split_num in range(splits):
        for inst_num in range(insts):
            midiToAudio(path_format.format(split_num, inst_num, "roll"),
                        path_format.format(split_num, inst_num, "art"),
                        save_path=save_format.format(split_num, inst_num))
            

def generate_synth(path_format, save_format, splits, insts=4):
    for split_num in range(splits):
        for inst_num in range(insts):
            midiToSynth(path_format.format(split_num, inst_num, "roll"),
                        path_format.format(split_num, inst_num, "art"),
                        save_path=save_format.format(split_num, inst_num))
            
def generate_audio_from_synth(path_format, splits, interpolation_rate=0.99, insts=4):
    for split_num in range(splits):
        for inst_num in range(insts):
            synthToAudio(path_format.format(split_num, inst_num),
                         interpolation_rate=interpolation_rate)