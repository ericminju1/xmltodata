# MusicXML parser

forked from https://github.com/qsdfo/musicxml_parser  
and added some more articulations

## Usage

    from xmlToMidi import xmlToMidi
    
    paths = ["Vn1_path.musicxml",
             "Vn2_path.musicxml",
             "Va_path.musicxml",
             "Vc_path.musicxml"]
    xmlToMidi(paths, save_path="path/to/file/without/.mid")

&nbsp;
REQUIRED: manually add tempo mark for splitting (Use any DAW program)

&nbsp;
split midi based on manual tempor marks

    from splitMidiData import splitMidiData
    
    splitMidiData("path/to/file/after/tempo")

&nbsp;
generate synthesis parameters before audio

    from generate_dataset import generate_synth

    path_format = "folder/after/split/{:02d}_{:01d}_{}.mid"
    save_format = "path/to/audio/{:02d}/{:01d}.npz"

    splits = 16  ## number of tempo changes
    generate_synth(path_format, save_format, splits=splits)

&nbsp;
generate audio from synth parameters

    from generate_dataset import generate_audio_from_synth

    path_format = "path/to/audio/{:02d}/{:01d}.npz"

    splits = 16  ## number of tempo changes
    generate_audio_from_synth(path_format, splits, interpolation_rate=0.99)
