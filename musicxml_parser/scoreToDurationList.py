#!/usr/bin/env python

import numpy as np
import xml.sax
import re
import os
import tempfile
from .smooth_dynamic import smooth_dyn
from .totalLengthHandler import TotalLengthHandler

mapping_step_midi = {
    'C': 0,
    'D': 2,
    'E': 4,
    'F': 5,
    'G': 7,
    'A': 9,
    'B': 11
}



class ScoreToPianorollHandler(xml.sax.ContentHandler):
    def __init__(self, division, total_length, number_pitches=128, discard_grace=False):
        self.CurrentElement = u""
        self.CurrentAttributes = {}
        # Instrument
        self.number_pitches = number_pitches
        self.identifier = u""                # Current identifier
        # and instrument name in the produced pianoroll
        self.content = u""
        self.part_instru_mapping = {}
        self.current_tempo = 0

        # Measure informations
        self.time = 0              # time counter
        self.division_score = -1     # rhythmic quantization of the original score (in division of the quarter note)
        self.division_pianoroll = division  # thythmic quantization of the pianoroll we are writting
        self.beat = -1
        self.beat_type = -1
        self.bar_length = -1
        self.measure_number = 0

        # Current note information
        # Pitch
        self.pitch_set = False
        self.step = u""
        self.step_set = False
        self.octave = 0
        self.octave_set = False
        self.alter = 0
        # Is it a rest ?
        self.rest = False
        # Is it a chord ? (usefull for the staccati)
        self.chord = False
        # Time
        self.duration = 0
        self.duration_set = False
        # Voices are used for the articulation
        self.current_voice = u""
        self.voice_set = False
        # Deal with grace notes
        self.grace = False
        self.slash = False
        self.discard_grace = discard_grace

        # Pianoroll
        self.total_length = total_length
        self.pianoroll = {}
        self.pianoroll_local = np.zeros([self.total_length * self.division_pianoroll, self.number_pitches], dtype=np.int)

        # Stop flags
        self.articulation = {}
        self.articulation_local = np.zeros([self.total_length * self.division_pianoroll, self.number_pitches], dtype=np.int)
        # Tied notes (not phrasing)
        self.tie_type = None
        self.slur_type = None
        self.tying = {}  # Contains voice -> tie_on?
        self.sluring = {}
        self.slur_note = None
        self.stop_slur = False
        # Staccati . Note that for chords the staccato tag is
        # ALWAYS on the first note of the chord if the file is correctly written
        self.previous_staccato = False
        self.staccato = False
        self.previous_staccatissimo = False
        self.staccatissimo = False

        # Time evolution of the dynamics
        self.writting_dynamic = False
        self.dynamics = np.zeros([total_length * division], dtype=np.float)
        self.dyn_flag = np.zeros([total_length * division], dtype=np.float)
        # Directions
        self.tempo_mark_ = False
        self.tempo_mark = True
        self.tempo_mark_count = 0
        # collect durations
        self.duration_list = {0:[]}

        ####################################################################
        ####################################################################
        ####################################################################

    def startElement(self, tag, attributes):
        self.CurrentElement = tag
        self.CurrentAttributes = attributes

        # Part information
        if tag == u"score-part":
            self.identifier = attributes[u'id']
        if tag == u"part":
            self.identifier = attributes[u'id']
            # And set to zeros time information
            self.time = 0
            self.division_score = -1
            # Initialize the pianoroll
            # Check if this instrument has already been seen
            self.pianoroll_local = np.zeros([self.total_length * self.division_pianoroll, self.number_pitches], dtype=np.int)
            # Initialize the articulations
            self.tie_type = None
            self.tying = {}  # Contains {voice -> tie_on} ?
            self.slur_type = None
            self.sluring = {}
            self.articulation_local = np.zeros([self.total_length * self.division_pianoroll, self.number_pitches], dtype=np.int)
            # Initialize the dynamics
            self.dynamics = np.zeros([self.total_length * self.division_pianoroll], dtype=np.float) + 0.5  # Don't initialize to zero knowing if no dynamic is given
            self.dyn_flag = {}

        if tag == u'measure':
            self.measure_number = attributes[u'number']
            if self.measure_number in ("0","1"):
                self.tempo_mark_ = True
                self.tempo_mark_count = 1
            elif (u'implicit' in attributes) and (attributes[u'implicit'] == "yes"):
                self.tempo_mark_ = True
                self.tempo_mark_count = 1
            else:
                if self.tempo_mark_count <= 0:
                    self.tempo_mark_ = False
                else:
                    self.tempo_mark_count -= 1
        
        if tag == u'barline':
            self.tempo_mark_ = True
            self.tempo_mark_count = 1
        
        if tag == u'note':
            self.not_played_note = False
            if u'print-object' in attributes.keys():
                if attributes[u'print-object'] == "no":
                    self.not_played_note = True
        if tag == u'rest':
            self.rest = True
        if tag == u'chord':
            if self.duration_set:
                raise NameError('A chord tag should be placed before the duration tag of the current note')
            self.time -= self.duration
            self.chord = True
        if tag == u'grace':
            self.grace = True
            self.slash = True if (u'slash' in attributes and attributes[u'slash']=="yes") else False # error

        ####################################################################
        if tag == u'direction':
            if u'placement' in attributes.keys() and (attributes[u'placement'] == 'above') and self.tempo_mark_:
                self.tempo_mark = True
        ####################################################################

        ###################################################################
        ###################################################################
        ###################################################################

    def endElement(self, tag):
        if tag == u'pitch':
            if self.octave_set and self.step_set:
                self.pitch_set = True
            self.octave_set = False
            self.step_set = False

        if tag == u'direction':
            self.tempo_mark = False
        
        if tag == u"note":
            # When leaving a note tag, write it in the pianoroll
            if not self.duration_set:
                if not self.grace:
                    raise NameError("XML misformed, a Duration tag is missing")

            not_a_rest = not self.rest
            not_a_grace = not self.grace or not self.discard_grace
            note_played = not self.not_played_note
            if not_a_rest and not_a_grace and note_played:
                # Check file integrity
                if not self.pitch_set:
                    print("XML misformed, a Pitch tag is missing")
                    return
                # Start and end time for the note
                start_time = int(self.time * self.division_pianoroll / self.division_score)
                if self.grace:
                    if self.slash:
                        end_time = start_time
                        # A grace note is an anticipation
                        start_time -= 1
                    else:
                        end_time = start_time
                        # A grace note is an anticipation
                        start_time -= 1
                else:
                    end_time = int((self.time + self.duration) * self.division_pianoroll / self.division_score)
                # Its pitch
                midi_pitch = mapping_step_midi[self.step] + self.octave * 12 + self.alter
                self.duration_list[self.current_tempo].append(self.duration)
                # Write it in the pianoroll
                # self.pianoroll_local[start_time:end_time, midi_pitch] = int(1)

                voice = u'1'
                if self.voice_set:
                    voice = self.current_voice

                # Initialize if the voice has not been seen before
                if voice not in self.tying:
                    self.tying[voice] = False

                if voice not in self.sluring:
                    self.sluring[voice] = False

                # Note that tying[voice] can't be set when opening the tie tag since
                # the current voice is not knew at this time
                if self.tie_type == u"start":
                    # Allows to keep on the tying if it spans over several notes
                    self.tying[voice] = True
                if self.tie_type == u"stop":
                    self.tying[voice] = False


                if self.stop_slur:
                    self.slur_note = None
                    self.stop_slur = False

                if self.slur_type == u"start":
                    # Allows to keep on the tying if it spans over several notes
                    self.sluring[voice] = True
                    if self.slur_note == None:
                        self.slur_note = midi_pitch
                if self.slur_type == u"stop":
                    self.sluring[voice] = False
                    self.stop_slur = True
                

                tie = self.tying[voice]
                slur = self.sluring[voice]

                # Staccati
                if self.chord:
                    self.staccato = self.previous_staccato
                    self.staccatissimo = self.previous_staccatissimo
                staccato = self.staccato
                staccatissimo = self.staccatissimo

                length = int(self.duration * self.division_pianoroll / self.division_score)
                # self.division_score=48 (=4분음표길이)
                # length_of_8 = int(self.division_pianoroll / 2)
                # length_of_16 = int(self.division_pianoroll / 4)
                # length_of_32 = int(self.division_pianoroll / 8)
                # length_of_64 = int(self.division_pianoroll / 16)

                if (not tie) and (not self.slur_note) and (not staccato) and (not staccatissimo):
                    self.articulation_local[start_time:end_time - 1, midi_pitch] = int(1)
                elif self.slur_note:
                    self.articulation_local[start_time:end_time, self.slur_note] = int(1)
                elif staccato:
                    stop_time = start_time + (end_time-start_time) // 2
                    self.articulation_local[start_time:stop_time, midi_pitch] = int(1)
                elif staccatissimo:
                    stop_time = start_time + (end_time-start_time) // 4
                    self.articulation_local[start_time:stop_time, midi_pitch] = int(1)
                elif tie:
                    self.articulation_local[start_time:end_time, midi_pitch] = int(1)
                

                if (not tie) and (not slur) and (not staccato) and (not staccatissimo):
                    self.pianoroll_local[start_time:end_time - 1, midi_pitch] = int(1)
                elif staccato:
                    stop_time = start_time + (end_time-start_time) // 2
                    self.pianoroll_local[start_time:stop_time, midi_pitch] = int(1)
                elif staccatissimo:
                    stop_time = start_time + (end_time-start_time) // 4
                    self.pianoroll_local[start_time:stop_time, midi_pitch] = int(1)
                elif tie:
                    self.pianoroll_local[start_time:end_time, midi_pitch] = int(1)
                elif slur:
                    self.pianoroll_local[start_time:end_time, midi_pitch] = int(1)
            

            # Increment the time counter
            if not self.grace and note_played:
                self.time += self.duration
            # Set to "0" different values
            self.pitch_set = False
            self.duration_set = False
            self.alter = 0
            self.rest = False
            self.grace = False
            self.voice_set = False
            self.tie_type = None
            self.previous_staccato = self.staccato
            self.staccato = False
            self.previous_staccatissimo = self.staccatissimo
            self.staccatissimo = False
            self.chord = False
            self.trill = False
            self.trill_next_step = 0

        if tag == u'backup':
            if not self.duration_set:
                raise NameError("XML Duration not set for a backup")
            self.time -= self.duration
            self.duration_set = False

        if tag == u'forward':
            if not self.duration_set:
                raise NameError("XML Duration not set for a forward")
            self.time += self.duration
            self.duration_set = False
            
        if tag == u'part-name':
            this_instru_name = self.content
            # print((u"@@ " + self.content + u"   :   " + this_instru_name).encode('utf8'))
            self.content = u""
            self.part_instru_mapping[self.identifier] = this_instru_name

        if tag == u'part':
            instru = self.part_instru_mapping[self.identifier]
            # Smooth the dynamics
            horizon = 4  # in number of quarter notes
            # dynamics = smooth_dyn(self.dynamics, self.dyn_flag, self.division_pianoroll, horizon)
            dynamics = self.dynamics
            # Apply them on the pianoroll and articulation
            if instru in self.pianoroll.keys():
                self.pianoroll[instru] = np.maximum(self.pianoroll[instru], np.transpose(np.multiply(np.transpose(self.pianoroll_local), dynamics)))
                self.articulation[instru] = np.maximum(self.articulation[instru], np.transpose(np.multiply(np.transpose(self.articulation_local), dynamics)))
            else:
                self.pianoroll[instru] = np.transpose(np.multiply(np.transpose(self.pianoroll_local), dynamics))
                self.articulation[instru] = np.transpose(np.multiply(np.transpose(self.articulation_local), dynamics))
        return
        
    def characters(self, content):
        # print(self.CurrentElement, self.CurrentElement == u"dashes")
        # print(content)
        # Avoid breaklines and whitespaces
        if content.strip():
            if self.CurrentElement == u"fifths":
                self.keysign = int(content)
            # Time and measure informations
            if self.CurrentElement == u"divisions":
                self.division_score = int(content)
                # print(self.division_score)
                if (not self.beat == -1) and (not self.beat_type == -1):
                    self.bar_length = int(self.division_score * self.beat * 4 / self.beat_type)
            if self.CurrentElement == u"beats":
                self.beat = int(content)
            if self.CurrentElement == u"beat-type":
                self.beat_type = int(content)
                assert (not self.beat == -1), "beat and beat type wrong"
                assert (not self.division_score == -1), "division non defined"
                self.bar_length = int(self.division_score * self.beat * 4 / self.beat_type)

            # Note informations
            if self.CurrentElement == u"duration":
                self.duration = int(content)
                self.duration_set = True
                if self.rest:
                    # A lot of (bad) publisher use a semibreve rest to say "rest all the bar"
                    if self.duration > self.bar_length:
                        self.duration = self.bar_length
            if self.CurrentElement == u"step":
                self.step = content
                self.step_set = True
            if self.CurrentElement == u"octave":
                self.octave = int(content)
                self.octave_set = True
            if self.CurrentElement == u"alter":
                if content == '-':
                    self.alter = -1
                    print("Alter problem")
                else:     
                    self.alter = int(content)
            if self.CurrentElement == u"voice":
                self.current_voice = content
                self.voice_set = True

            if self.CurrentElement == u"part-name":
                self.content += content

            ############################################################
            # Directions
            if self.CurrentElement == u'words':
                if self.tempo_mark:
                    t_start = int(self.time * self.division_pianoroll / self.division_score)
                    if any(c.isupper() for c in content):
                        print(content, t_start)
                        self.current_tempo = t_start
                        self.duration_list[self.current_tempo] = []
                        self.tempo_mark = False
        return
        

def search_re_list(string, expression):
    for value in expression:
        result_re = re.search(value, string, flags=re.IGNORECASE | re.UNICODE)
        if result_re is not None:
            return True
    return False


def pre_process_file(file_path):
    """Simply consists in removing the DOCTYPE that make the xml parser crash
    
    """

    temp_file = tempfile.NamedTemporaryFile('w', suffix='.xml', prefix='tmp', delete=False, encoding='utf-8')
    
    # Remove the doctype line
    with open(file_path, 'r', encoding="utf-8") as fread:
        for line in fread:
            if not re.search(r'<!DOCTYPE', line):
                temp_file.write(line)                
    temp_file_path = temp_file.name
    temp_file.close()
    
    # Return the new path
    return temp_file_path

    
def scoreToPianorolld(score_path, quantization):
    # Remove DOCTYPE
    tmp_file_path = pre_process_file(score_path)

    # Get the total length in quarter notes of the track
    pre_parser = xml.sax.make_parser()
    pre_parser.setFeature(xml.sax.handler.feature_namespaces, 0)
    Handler_length = TotalLengthHandler()
    pre_parser.setContentHandler(Handler_length)
    pre_parser.parse(tmp_file_path)
    total_length = Handler_length.total_length
    # Float number
    total_length = int(total_length)

    # Now parse the file and get the pianoroll, articulation and dynamics
    parser = xml.sax.make_parser()
    parser.setFeature(xml.sax.handler.feature_namespaces, 0)
    Handler_score = ScoreToPianorollHandler(quantization, total_length)
    parser.setContentHandler(Handler_score)
    parser.parse(tmp_file_path)

    # Using Mapping, build concatenated along time and pitch pianoroll
    pianoroll = {}
    articulation = {}
    for instru_name, mat in Handler_score.pianoroll.items():
        pianoroll[instru_name] = (mat*128).astype(int)
    for instru_name, mat in Handler_score.articulation.items():
        articulation[instru_name] = mat
    
    os.remove(tmp_file_path)
    return pianoroll, articulation, Handler_score

if __name__ == '__main__':
    score_path = "/Users/leo/Recherche/GitHub_Aciditeam/database/Arrangement/SOD/OpenMusicScores/0/Belle qui tiens ma vie   - Arbeau, Toinot .xml"
    quantization = 8
    pianoroll, articulation = scoreToPianorolld(score_path, quantization)