from midi_ddsp import load_pretrained_model
from midi_ddsp.utils.midi_synthesis_utils import synthesize_mono_midi
from midi_ddsp.utils.inference_utils import get_process_group
import numpy as np
import pretty_midi
import os

synthesis_generator, expression_generator = load_pretrained_model()

def midiToSynth(roll_path, art_path, interpolation_rate=0.99, save_path=None):
    part_num = int(roll_path.split("/")[-1][-10])
    part_to_inst_dict = {0:0, 1:0, 2:1, 3:2}
    instrument_id = part_to_inst_dict[part_num]
    midi_audio, midi_control_params, midi_synth_params, conditioning_df = \
        synthesize_mono_midi(synthesis_generator, expression_generator,
                            roll_path, instrument_id, output_dir=None, 
                            pitch_offset=0, speed_rate=1)

    f0_ori = midi_synth_params['f0_hz']
    amps_ori = midi_synth_params['amplitudes'].numpy()[0,...,0]
    noise_ori = midi_synth_params['noise_magnitudes'].numpy()
    hd_ori = midi_synth_params['harmonic_distribution']

    midi_audio2, midi_control_params2, midi_synth_params2, conditioning_df2 = \
        synthesize_mono_midi(synthesis_generator, expression_generator,
                            art_path, instrument_id, output_dir=None, 
                            pitch_offset=0, speed_rate=1)

    amps_new = midi_synth_params2['amplitudes'].numpy()[0,...,0]
    noise_new = midi_synth_params2['noise_magnitudes'].numpy()

    new_length = min(amps_ori.shape[0], amps_new.shape[0])

    ## Dynamic
    mid0 = pretty_midi.PrettyMIDI(roll_path)
    expression_dict = {0:80}
    for cc in mid0.instruments[0].control_changes:
        if cc.number == 11:
            val = cc.value
            expression_dict[cc.time*250] = val
    expression_dict[1e16] = val

    expression_times = sorted(list(expression_dict.keys()))+[1e16]
    cursor = 0
    expression_list = []
    for i in range(new_length):
        while (i >= expression_times[cursor+1]):
            cursor += 1
        k0 = expression_times[cursor]
        v0 = expression_dict[k0]
        k1 = expression_times[cursor+1]
        v1 = expression_dict[k1]
        if np.abs(v0-v1) <= 2:
            w0 = (k1-i) / (k1-k0)
            w1 = (i-k0) / (k1-k0)
            expression_list.append(w0*v0+w1*v1)
        else:
            expression_list.append(v0)

    expression = np.array(expression_list)
    expression = (expression+1)/128
    

    if save_path is None:
        save_path = "{}".format(roll_path[:-9])
    
    np.savez(save_path+".npz", 
             f0_ori=f0_ori, 
             amps_ori=amps_ori,
             noise_ori=noise_ori,
             hd_ori=hd_ori,
             amps_new=amps_new,
             noise_new=noise_new,
             expression=expression)