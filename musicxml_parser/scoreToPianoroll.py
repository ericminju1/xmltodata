#!/usr/bin/env python
# -*- coding: utf8 -*-

# Class to parse a MusicXML score into a pianoroll
# Input :
#       - division : the desired quantization in the pianoroll
#       - instru_dict : a dictionary
#       - total_length : the total length in number of quarter note of the parsed file
#            (this information can be accessed by first parsing the file with the durationParser)
#       - discard_grace : True to discard grace notes and not write them in the pianoroll
#
# A few remarks :
#       - Pianoroll : the different instrument are mapped from the name written in part-name
#           to a unique name. This mapping is done through regex indexed in instru_dict.
#           This dictionary is imported through the json format.
#

import xml.sax
import re
import os
import tempfile
import numpy as np
from smooth_dynamic import smooth_dyn
from totalLengthHandler import TotalLengthHandler

mapping_step_midi = {
    'C': 0,
    'D': 2,
    'E': 4,
    'F': 5,
    'G': 7,
    'A': 9,
    'B': 11
}

next_step = {
    'C': 'D',
    'D': 'E',
    'E': 'F',
    'F': 'G',
    'G': 'A',
    'A': 'B',
    'B': 'C'
}

keysign_notes = ['F','C','G','D','A','E','B']

mapping_dyn_number = {
    # Value drawn from http://www.wikiwand.com/en/Dynamics_%28music%29
    'ppp': 0.125,
    'pp': 0.258,
    'p': 0.383,
    'mp': 0.5,
    'mf': 0.625,
    'f': 0.75,
    'ff': 0.875,
    'fff': 0.984
}


class ScoreToPianorollHandler(xml.sax.ContentHandler):
    def __init__(self, division, total_length, shortest_notes, number_pitches=128, discard_grace=False):
        self.CurrentElement = u""
        self.CurrentAttributes = {}
        # Instrument
        self.number_pitches = number_pitches
        self.identifier = u""                # Current identifier
        # and instrument name in the produced pianoroll
        self.content = u""
        self.part_instru_mapping = {}

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
        self.direction_type = None
        self.direction_start = None
        self.direction_stop = None
        self.cresc = 0
        self.cresc_start_dyn = None
        self.cresc_nowedge = False
        self.wait_for_dyn = False
        self.wait_for_dyn_time = 0
        self.dash_time = 0
        self.dash = False
        # key signature
        self.keysign = 0
        # trill
        self.trill = False
        self.trill_new_step = 0
        self.trill_new_octave = 0
        self.trill_new_alter = 0
        self.trill_next_step = 0
        self.trill_length = 0
        self.shortest_notes = shortest_notes

        # tremolo
        self.tremolo = False
        self.tremolo_double = False
        self.tremolo_length = 0
        self.tremolo_start_pitch = 0
        self.tremolo_start_time = 0
        self.tremolo_end = False
        self.time_mod = False
        self.actual_notes = 0
        self.normal_notes = 0

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

        time_pianoroll = int(self.time * self.division_pianoroll / self.division_score)
        ####################################################################
        if tag == u'tie':
            self.tie_type = attributes[u'type']

        if tag == u'slur':
            self.slur_type = attributes[u'type']

        if tag == u'staccato':
            self.staccato = True

        if tag == u'staccatissimo':
            self.staccatissimo = True
        
        ####################################################################
        # Trills
        if tag == u'trill-mark':
            self.trill = True
            midi_pitch = mapping_step_midi[self.step] + self.octave * 12 + self.alter
            self.trill_new_step = next_step[self.step]
            self.trill_new_octave = self.octave + 1 if self.step=='B' else self.octave
            self.trill_new_alter = 0
            if self.keysign >= 0:
                alter_keys = keysign_notes[:self.keysign]
                if self.trill_new_step in alter_keys:
                    self.trill_new_alter = 1
            else:
                alter_keys = keysign_notes[self.keysign:]
                if self.trill_new_step in alter_keys:
                    self.trill_new_alter = -1
            self.trill_next_step = mapping_step_midi[self.trill_new_step] + self.trill_new_octave * 12 + self.trill_new_alter
            keys = np.array(list(self.shortest_notes.keys()))
            tempo_key = max(sorted(keys[keys<=time_pianoroll]))
            self.trill_length = self.shortest_notes[tempo_key] * 1/2
            self.trill_length = int(self.trill_length * self.division_pianoroll / self.division_score)

        if tag == u'time-modification':
            self.time_mod = True

        ####################################################################
        # Dynamics
        dyn = False
        if tag in mapping_dyn_number:
            self.dynamics[time_pianoroll:] = mapping_dyn_number[tag]
            self.dyn_flag[time_pianoroll] = 'N'
            dyn = True
        elif tag in (u"sf", u"sfz", u"sffz", u"fz"):
            self.dynamics[time_pianoroll] = mapping_dyn_number[u'fff']
            self.dyn_flag[time_pianoroll] = 'N'
            dyn = True
        elif tag == u'fp':
            self.dynamics[time_pianoroll] = mapping_dyn_number[u'f']
            self.dynamics[time_pianoroll + 1:] = mapping_dyn_number[u'p']
            self.dyn_flag[time_pianoroll] = 'N'
            self.dyn_flag[time_pianoroll + 1] = 'N'
            dyn = True
        elif tag == u'ffp':
            self.dynamics[time_pianoroll] = mapping_dyn_number[u'ff']
            self.dynamics[time_pianoroll + 1:] = mapping_dyn_number[u'p']
            self.dyn_flag[time_pianoroll] = 'N'
            self.dyn_flag[time_pianoroll + 1] = 'N'
            dyn = True

        # for crescendo 
        if self.cresc != 0 and self.wait_for_dyn and dyn:
            if self.cresc_nowedge:
                self.direction_stop = time_pianoroll
                self.cresc_nowedge = False

            starting_dyn = self.cresc_start_dyn 
            ending_dyn = self.dynamics[time_pianoroll]
            temp_ending_dyn = ending_dyn.copy()
            if self.cresc > 0:
                if ending_dyn <= starting_dyn:
                    temp_ending_dyn = min(starting_dyn + 0.25, 1)
                self.dyn_flag[self.direction_start] = 'Cresc_start'
                self.dyn_flag[self.direction_stop] = 'Cresc_stop'
            if self.cresc < 0:
                if ending_dyn >= starting_dyn:
                    temp_ending_dyn = max(starting_dyn - 0.25, 0)
                self.dyn_flag[self.direction_start] = 'Dim_start'
                self.dyn_flag[self.direction_stop] = 'Dim_stop'

            self.dynamics[self.direction_start:self.direction_stop] = \
                np.linspace(starting_dyn, temp_ending_dyn, self.direction_stop - self.direction_start)
            # print(starting_dyn, temp_ending_dyn, ending_dyn, self.direction_start, self.direction_stop)
            # self.dynamics[self.direction_stop:] = ending_dyn
            self.direction_start = None
            self.direction_stop = None
            self.cresc = 0
            self.wait_for_dyn = False

        elif self.cresc != 0 and self.wait_for_dyn and (not self.dash) \
            and (time_pianoroll > self.wait_for_dyn_time + 4 * self.division_pianoroll): # 4 quarter notes
            if self.cresc_nowedge:
                self.direction_stop = time_pianoroll
                self.cresc_nowedge = False
            
            starting_dyn = self.cresc_start_dyn 
            if self.cresc > 0:
                ending_dyn = min(starting_dyn + 0.25, 1)
                self.dyn_flag[self.direction_start] = 'Cresc_start'
                self.dyn_flag[self.direction_stop] = 'Cresc_stop'
            else:
                ending_dyn = max(starting_dyn - 0.25, 0)
                self.dyn_flag[self.direction_start] = 'Dim_start'
                self.dyn_flag[self.direction_stop] = 'Dim_stop'
            
            # print(self.direction_start, self.direction_stop, self.measure_number)
            self.dynamics[self.direction_start:self.direction_stop] = \
                    np.linspace(starting_dyn, ending_dyn, self.direction_stop - self.direction_start)
            # print(starting_dyn, ending_dyn, ending_dyn, self.direction_start, self.direction_stop, "passed 4 notes")
            self.dynamics[self.direction_stop:] = ending_dyn
            self.direction_start = None
            self.direction_stop = None
            self.cresc = 0
            self.wait_for_dyn = False

        # Directions
        # Cresc end dim are written with an arbitrary slope, then adjusted after the file
        # has been parsed by a smoothing function
        if tag == u'wedge':
            if attributes[u'type'] in (u'diminuendo', u'crescendo'):
                if self.cresc:
                    self.direction_stop = time_pianoroll
                    if self.cresc_nowedge:
                        self.cresc_nowedge = False
                    starting_dyn = self.cresc_start_dyn 
                    ending_dyn = self.dynamics[time_pianoroll]
                    temp_ending_dyn = ending_dyn.copy()
                    if self.cresc > 0:
                        if ending_dyn <= starting_dyn:
                            temp_ending_dyn = min(starting_dyn + 0.25, 1)
                        self.dyn_flag[self.direction_start] = 'Cresc_start'
                        self.dyn_flag[self.direction_stop] = 'Cresc_stop'
                    if self.cresc < 0:
                        if ending_dyn >= starting_dyn:
                            temp_ending_dyn = max(starting_dyn - 0.25, 0)
                        self.dyn_flag[self.direction_start] = 'Dim_start'
                        self.dyn_flag[self.direction_stop] = 'Dim_stop'
                    self.dynamics[self.direction_start:self.direction_stop] = \
                        np.linspace(starting_dyn, temp_ending_dyn, self.direction_stop - self.direction_start)
                    if self.cresc < 0 and attributes[u'type'] == u'crescendo':
                        self.dynamics[self.direction_stop:] = temp_ending_dyn
                    if self.cresc > 0 and attributes[u'type'] == u'diminuendo':
                        self.dynamics[self.direction_stop:] = temp_ending_dyn
                    self.direction_start = None
                    self.direction_stop = None
                    self.cresc = 0
                    self.wait_for_dyn = False

                self.direction_start = time_pianoroll
                self.direction_type = attributes[u'type']
                if attributes[u'type'] == u'diminuendo':
                    self.cresc = -1
                else:
                    self.cresc = 1
            elif attributes[u'type'] == u'stop':
                self.direction_stop = time_pianoroll
                if self.duration_set:
                    self.direction_stop += self.duration * self.division_pianoroll / self.division_score
                if self.direction_start is None:
                    pass
                    # raise NameError('Stop flag for a direction, but no direction has been started')
                else:
                    starting_dyn = self.dynamics[self.direction_start]
                    self.cresc_start_dyn  = starting_dyn
                    self.wait_for_dyn = True
                    self.wait_for_dyn_time = self.direction_stop

        ###################################################################
        ###################################################################
        ###################################################################

    def endElement(self, tag):
        if tag == u'pitch':
            if self.octave_set and self.step_set:
                self.pitch_set = True
            self.octave_set = False
            self.step_set = False

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
                    # self.stop_slur = True
                

                tie = self.tying[voice]
                slur = self.sluring[voice]

                if (not tie) and (not slur):
                    self.stop_slur = True
                    # self.slur_note = None

                # Staccati
                if self.chord:
                    self.staccato = self.previous_staccato
                    self.staccatissimo = self.previous_staccatissimo
                staccato = self.staccato
                staccatissimo = self.staccatissimo


                # if self.measure_number == "39":
                #     print(start_time, self.step)

                
                ########################################################################################################
                # articulation
                if self.tremolo and self.time_mod and self.tremolo_end:
                    if self.actual_notes / self.normal_notes == 2.:
                        pass
                    else:
                        self.tremolo_length = self.tremolo_length * self.normal_notes / self.actual_notes 
                        self.time_mod = False
                        self.normal_notes = 0
                        self.actual_notes = 0

                if self.tremolo and (not self.tremolo_double):
                    if self.tremolo_end:
                        art_end_time = end_time - 1
                        intervals = np.arange(self.tremolo_start_time, art_end_time, self.tremolo_length, dtype=int)
                        for i in range(0, len(intervals)):
                            s = intervals[i]
                            e = intervals[i+1] if i < len(intervals)-1 else s + int(self.tremolo_length)
                            self.articulation_local[s:e-1, midi_pitch] = int(1)

                elif (not tie) and (not self.slur_note) and (not staccato) and (not staccatissimo):
                    art_end_time = end_time - 1
                    self.articulation_local[start_time:art_end_time, midi_pitch] = int(1)
                elif self.slur_note and (not tie) and (not slur):
                    art_end_time = end_time - 1
                    self.articulation_local[start_time:art_end_time, self.slur_note] = int(1)
                elif self.slur_note:
                    art_end_time = end_time
                    self.articulation_local[start_time:art_end_time, self.slur_note] = int(1)
                elif staccato:
                    art_end_time = start_time + (end_time-start_time) // 2
                    self.articulation_local[start_time:art_end_time, midi_pitch] = int(1)
                elif staccatissimo:
                    art_end_time = start_time + (end_time-start_time) // 4
                    self.articulation_local[start_time:art_end_time, midi_pitch] = int(1)
                elif tie:
                    art_end_time = end_time
                    self.articulation_local[start_time:art_end_time, midi_pitch] = int(1)
                
                ########################################################################################################
                # pianoroll  
                      
                if self.trill:
                    delay_start = int(np.random.random()*self.trill_length*0.7) if np.random.random()<=0.5 else 0
                    intervals = np.arange(start_time+delay_start, art_end_time, self.trill_length, dtype=int)[1:]
                    e = start_time
                    for i in range(0, len(intervals)-1, 2):
                        s = intervals[i-1] if i > 0 else start_time
                        m = intervals[i]
                        e = intervals[i+1]
                        self.pianoroll_local[s:m, midi_pitch] = int(1)
                        self.pianoroll_local[m:e, self.trill_next_step] = int(1)
                    self.pianoroll_local[e:art_end_time,midi_pitch] = int(1)

                elif self.tremolo:
                    if self.tremolo_end:
                        if self.tremolo_double:
                            intervals = np.arange(self.tremolo_start_time, art_end_time, self.tremolo_length, dtype=int)[1:]
                            e = self.tremolo_start_time
                            for i in range(0, len(intervals), 2):
                                s = intervals[i-1] if i > 0 else self.tremolo_start_time
                                m = intervals[i]
                                e = intervals[i+1] if i < len(intervals)-1 else 2*m-s
                                if e > art_end_time:
                                    e = art_end_time
                                self.pianoroll_local[s:m, self.tremolo_start_pitch] = int(1)
                                self.pianoroll_local[m:e, midi_pitch] = int(1)
                            self.pianoroll_local[e:art_end_time,midi_pitch] = int(1)
                            self.tremolo_start_time
                            self.tremolo_length
                        else:
                            intervals = np.arange(self.tremolo_start_time, art_end_time, self.tremolo_length, dtype=int)
                            for i in range(0, len(intervals)):
                                s = intervals[i]
                                e = intervals[i+1] if i < len(intervals)-1 else s + int(self.tremolo_length)
                                self.pianoroll_local[s:e-1, midi_pitch] = int(1)
                        self.tremolo = False
                        self.tremolo_double = False
                        self.tremolo_length = 0
                        self.tremolo_start_pitch = 0
                        self.tremolo_start_time = 0
                        self.tremolo_end = False
                        
                elif (not tie) and (not slur) and (not staccato) and (not staccatissimo):
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
            self.trill_new_step = 0
            self.trill_new_octave = 0
            self.trill_new_alter = 0
            self.trill_next_step = 0
            self.trill_length = 0
            self.time_mod = False
            self.actual_notes = 0
            self.normal_notes = 0

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
            
            # for trill mark
            if self.CurrentElement == u'accidental-mark':
                if self.trill:
                    accidental_map = {'natural':0, 'sharp':1, 'double-sharp':2, 'flat':-1, 'flat-flat':-2}
                    self.trill_new_alter = accidental_map[self.content]
                    self.trill_next_step = mapping_step_midi[self.trill_new_step] + self.trill_new_octave * 12 + self.trill_new_alter

            # tremolo
            if self.CurrentElement == u'tremolo':
                self.tremolo = True
                type = self.CurrentAttributes[u'type']
                if type == 'single':
                    self.tremolo_double = False
                    self.tremolo_end = True
                    self.tremolo_start_time = int(self.time * self.division_pianoroll / self.division_score)
                elif type == 'start':
                    self.tremolo_double = True
                    self.tremolo_start_pitch =  mapping_step_midi[self.step] + self.octave * 12 + self.alter
                    self.tremolo_start_time = int(self.time * self.division_pianoroll / self.division_score)
                elif type == 'stop':
                    self.tremolo_end = True
                    
                self.tremolo_length = self.division_pianoroll / (2**int(content))
            
            if self.time_mod:
                if self.CurrentElement == u'actual-notes':
                    self.actual_notes = int(content)
                if self.CurrentElement == u'normal-notes':
                    self.normal_notes = int(content)

            ############################################################
            # Directions
            if self.CurrentElement == u'words':
                # We consider that a written dim or cresc approximately span over 4 quarter notes.
                # If its less its gonna be overwritten by the next dynamic
                is_cresc = re.match(r'[cC]res.*', content)
                is_decr = re.match(r'[dD]ecres.*', content)
                is_dim = re.match(r'[dD]im.*', content)
                
                if is_dim or is_cresc or is_decr:
                    t_start = int(self.time * self.division_pianoroll / self.division_score)
                    t_end = t_start + 4 * self.division_pianoroll
                    start_dyn = self.dynamics[t_start]
                    # if is_cresc:
                    #     self.dynamics[t_start:t_end] = np.linspace(start_dyn, min(start_dyn + 0.25, 1), t_end - t_start)
                    # elif is_dim or is_decr:
                    #     self.dynamics[t_start:t_end] = np.linspace(start_dyn, max(start_dyn - 0.25, 0), t_end - t_start)
                    if is_cresc:
                        self.cresc = 1
                        self.cresc_start_dyn = start_dyn
                        self.direction_start = t_start
                        self.wait_for_dyn = True
                        self.wait_for_dyn_time = t_start
                        self.cresc_nowedge = True
                    elif is_dim or is_decr:
                        self.cresc = -1
                        self.cresc_start_dyn = start_dyn
                        self.direction_start = t_start
                        self.wait_for_dyn = True
                        self.wait_for_dyn_time = t_start
                        self.cresc_nowedge = True

        elif self.CurrentElement == u"dashes" and self.cresc and self.cresc_nowedge:
            if self.dash_time != self.time:
                self.dash_time = self.time
                if self.CurrentAttributes[u'type'] == "start":
                    self.dash = True
                elif self.CurrentAttributes[u'type'] == "stop":
                    self.dash = False
                    t_start = int(self.time * self.division_pianoroll / self.division_score)
                    self.direction_stop = t_start
                    self.wait_for_dyn_time = t_start
                    self.cresc_nowedge = False
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

    
def scoreToPianoroll(score_path, quantization, shortest_notes):
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
    Handler_score = ScoreToPianorollHandler(quantization, total_length, shortest_notes)
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
    pianoroll, articulation = scoreToPianoroll(score_path, quantization)