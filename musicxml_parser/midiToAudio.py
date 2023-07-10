from midi_ddsp import load_pretrained_model
from midi_ddsp.utils.midi_synthesis_utils import synthesize_mono_midi
from midi_ddsp.utils.inference_utils import get_process_group
import numpy as np
import soundfile as sf
import pretty_midi

synthesis_generator, expression_generator = load_pretrained_model()

def midiToAudio(roll_path, art_path, interpolation_rate=0.99, save_path=None):
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

    inter_amp = (1-interpolation_rate)*np.power(10,amps_ori[:new_length]) + \
                 interpolation_rate*np.power(10,amps_new[:new_length])
    amps_changed = np.log10(inter_amp)[np.newaxis,:,np.newaxis]

    inter_noise = (1-interpolation_rate)*np.power(10,noise_ori[:,:new_length,:]) + \
                   interpolation_rate*np.power(10,noise_new[:,:new_length,:])
    noise_changed = np.log10(inter_noise)

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
    expression = 3*np.log10(expression/128)

    amps_changed = amps_changed + expression[np.newaxis,:,np.newaxis]
    noise_changed = noise_changed + expression[np.newaxis,:,np.newaxis]

    processor_group = get_process_group(new_length, use_angular_cumsum=True)
    midi_audio_changed = processor_group({'amplitudes': amps_changed,
                            'harmonic_distribution': hd_ori[:,:new_length,:],
                            'noise_magnitudes': noise_changed,
                            'f0_hz': f0_ori[:,:new_length,:],},
                            verbose=False)

    if synthesis_generator.reverb_module is not None:
        midi_audio_changed = synthesis_generator.reverb_module(midi_audio_changed, reverb_number=instrument_id, training=False)
    
    final_audio = midi_audio_changed.numpy()

    if save_path is None:
        save_path = "{}.wav".format(roll_path[:-9])

    sf.write(save_path, final_audio[0], samplerate=16000)