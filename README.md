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

manually add tempo mark for splitting 

    from splitMidiData import splitMidiData
    
    splitMidiData("path/to/file/after/tempo")

&nbsp;

    from generate_dataset import generate_dataset

    path_format = "folder/after/split/{:02d}_{:01d}_{}.mid"
    save_format = "path/to/audio/{:02d}/{:01d}.wav"

    splits = 16  ## number of tempo changes
    generate_dataset(path_format, save_format, splits=splits)
