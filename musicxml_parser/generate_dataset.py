from midiToAudio import .midiToAudio

def generate_dataset(path_format, save_format, splits, insts=4):
    for split_num in range(splits):
        for inst_num in range(insts):
            midiToAudio(path_format.format(split_num, inst_num, "roll"),
                        path_format.format(split_num, inst_num, "art"),
                        save_path=save_format(split_num, inst_num))