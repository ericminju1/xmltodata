from midi_ddsp import load_pretrained_model
from midi_ddsp.utils.midi_synthesis_utils import synthesize_mono_midi
from midi_ddsp.utils.inference_utils import get_process_group
import numpy as np
import soundfile as sf
import os

synthesis_generator, expression_generator = load_pretrained_model()

def exp_sigmoid(x):
  y = 1 / (1+np.exp(-x))
  y = y ** np.log(10.0)
  y = y * 2.0
  return y

def rev_exp_sigmoid(y):
  x = y / 2.0
  x = x ** (1/np.log(10.0))
  x  = -np.log((1/x)-1)
  return x

def synthToAudio(synth_path, interpolation_rate=0.99, save_path=None):
    data = np.load(synth_path)

    part_num = int(synth_path.split("/")[-1][-5])
    part_to_inst_dict = {0:0, 1:0, 2:1, 3:2}
    instrument_id = part_to_inst_dict[part_num]

    f0_ori = data['f0_ori']
    amps_ori = data['amps_ori']
    noise_ori = data['noise_ori']
    hd_ori = data['hd_ori']

    amps_new = data['amps_new']
    noise_new = data['noise_new']

    new_length = min(amps_ori.shape[0], amps_new.shape[0])

    inter_amp = (1-interpolation_rate)*exp_sigmoid(amps_ori[:new_length]) + \
                 interpolation_rate*exp_sigmoid(amps_new[:new_length])
    amps_changed = rev_exp_sigmoid(inter_amp)[np.newaxis,:,np.newaxis]

    inter_noise = (1-interpolation_rate)*exp_sigmoid(noise_ori[:,:new_length,:]) + \
                   interpolation_rate*exp_sigmoid(noise_new[:,:new_length,:])
    noise_changed = rev_exp_sigmoid(inter_noise)

    expression = data['expression']
    log_expression = 2*np.log10(expression)

    amps_changed = amps_changed + log_expression[np.newaxis,:,np.newaxis] + 2
    noise_changed = noise_changed + log_expression[np.newaxis,:,np.newaxis] 

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
        save_path = "{}.wav".format(synth_path[:-4])

    directory = "/".join(save_path.split('/')[:-1])
    if not os.path.exists(directory):
        os.makedirs(directory)

    sf.write(save_path, final_audio[0], samplerate=16000)