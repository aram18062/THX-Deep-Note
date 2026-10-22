# THX Deep Note: Controls Reference (v2)

All three scripts use the same v2 synthesis engine and the same controls. With
the same settings, they produce identical audio.

| Script | What it does | Needs |
|---|---|---|
| `deep_note_play_v2.py` | Renders the audio, plays it through your speakers, and can also save `deep_note.wav` | numpy, scipy, sounddevice |
| `deep_note.py` | Renders the audio and writes `deep_note.wav` plus `deep_note.png` (waveform, voice trajectories, spectrogram) | numpy, scipy, matplotlib |
| `deep_note_wav.py` | Renders the audio and writes `deep_note.wav` only | numpy, scipy |

All controls are constants at the top of each file. If you change one, change
it in every script you use so they stay in sync.

---

## How the piece is built

Thirty bright, cello-like voices go through four phases:

1. **Wander** (about 0.5–12 s): voices enter one at a time and drift randomly
   around 140–270 Hz at a quiet, steady level.
2. **Dive** (about 12–17.6 s): every voice glides to its note in the final D
   chord. Voices headed for the bass dive first, voices headed for the treble
   soar upward, and a crescendo rises about 22 dB.
3. **Hold**: the full chord sounds at peak level.
4. **Release**: a gentle sag, with the treble fading faster than the bass,
   then a fade to silence.

The timeline is computed, not fixed:

```
arrival    = latest time any voice finishes its glide   (~17.6 s by default)
hold_end   = arrival + HOLD_TIME
fade_start = hold_end + SAG_TIME
end        = fade_start + FADE_OUT + TAIL_SILENCE      (~31.6 s by default)
```

---

## General

| Control | Default | What it does |
|---|---|---|
| `SAMPLE_RATE` | 96000 | Output sample rate in Hz. 48000 renders about twice as fast. |
| `NUM_VOICES` | 30 | Number of voices. The chord is defined for 30 voices. A lower number drops the top notes of the chord. **Values above 30 will fail** unless you add voices to the chord's `counts` list (see [Final chord](#final-chord)). |

## Phase 1: Wander

| Control | Default | What it does |
|---|---|---|
| `START_F_MIN`, `START_F_MAX` | 140, 270 Hz | Each voice starts on a random pitch in this range. |
| `WANDER_SEMITONES` | 1.0 | How far each voice's pitch drifts, in semitones (standard deviation). 0 holds each voice on a steady pitch. |
| `WANDER_RATE` | 0.25 | Random drift control points per second. Lower values give slower, lazier drift. Higher values make it more restless. |
| `ONSET_MIN`, `ONSET_MAX` | 0.5, 3.0 s | Each voice enters at a random time in this window, so the piece starts in silence and fills in. |
| `VOICE_FADE_IN` | 1.0 s | How long each voice takes to fade in after it enters. |
| `FADE_IN` | 6.5 s | Overall swell from silence up to the wander level. |
| `FADE_IN_POWER` | 1.5 | Shape of that swell. 1 is linear in amplitude. Values above 1 start more slowly. |
| `PLATEAU_DB` | −22 dB | Loudness of the wander section relative to the final peak. This is also the size of the crescendo. |

## Phase 2: Dive

Voices are ranked by their target note, from 0 (lowest, D1) to 1 (highest, A6).

| Control | Default | What it does |
|---|---|---|
| `GLIDE_START` | 11.5 s | When the lowest-bound voices begin to dive. |
| `GLIDE_STAGGER` | 2.5 s | Higher targets start up to this much later, scaled by rank. This creates the bass-first cascade. |
| `GLIDE_JITTER` | 0.5 s | Random extra delay added to each voice's start, including the upward voices. |
| `GLIDE_T_MIN`, `GLIDE_T_MAX` | 3.0, 4.5 s | Random glide length for each voice. These voices use a smooth ease-in/ease-out curve. |
| `CRESC_START` | 12.5 s | When the crescendo begins. It rises linearly in dB from `PLATEAU_DB` to 0 dB and peaks when the last voice lands. |

### The soaring upper voices

| Control | Default | What it does |
|---|---|---|
| `UP_THRESHOLD` | 400 Hz | Voices whose target is above this pitch use the upward-sweep settings below instead of the ones above. |
| `UP_GLIDE_START` | 12.0 s | When the upward voices take off. `GLIDE_JITTER` is added to each one. |
| `UP_GLIDE_T_MIN`, `UP_GLIDE_T_MAX` | 5.0, 5.5 s | Random length of each upward climb. The longest climb usually sets `arrival`. |
| `UP_GLIDE_POWER` | 1.3 | Acceleration of the climb. Values above 1 start slowly and finish fast. 1 is a steady glide in pitch. |

## Final chord

The chord is D1 A1 D2 A2 D3 A3 D4 A4 D5 A5 F#6 A6 in just intonation, with
`3 1 4 1 4 3 3 2 2 2 2 3` voices on those notes. To change the voicing, edit
the `ratios` and `counts` lists in `build_final_chord()`.

| Control | Default | What it does |
|---|---|---|
| `D1_FREQ` | 37.8 Hz | Pitch of the lowest note. Every other note is a ratio of it. 37.8 Hz matches the reference recording, which is about half a semitone sharp (A4 ≈ 453.6 Hz). For A4 = 440, use 36.667. |
| `DETUNE_BASS` | 10 cents | Voices sharing a note are spread evenly across ± this amount at the bottom of the chord. This makes the unisons beat every few seconds instead of slowly cancelling each other out. |
| `DETUNE_TOP` | 6 cents | The same spread at the top of the chord. It is tighter because F#6 is exposed. |
| `DETUNE_RANDOM` | 2 cents | Additional random detune for each voice (standard deviation). |
| `UPPER_FLAT_CENTS` | 12 cents | Lowers the upper chord by up to this much relative to pure ratios, ramping in over the first 3 octaves above D1. This matches the reference recording's tuning. |

## Balance

| Control | Default | What it does |
|---|---|---|
| `BASS_BOOST_DB` | 12 dB | Extra gain for voices that land on low notes. |
| `BASS_BOOST_FREQ` | 90 Hz | Crossover of the bass boost. Notes well below it get the full boost, and the boost falls off above it. Lower values favor D1. |
| `BOOST_RAMP_POWER` | 3.0 | The boost fades in as glide progress raised to this power. A diving voice therefore gets loud only near the end of its dive, instead of jumping out as soon as it starts. |
| `TREBLE_TILT_DB` | 4 dB | After landing, upper voices settle this much quieter, scaled by rank. The climb itself stays at full strength. |
| `TILT_TIME` | 2.0 s | How long that settling takes after `arrival`. |

## Timbre

Each voice is a sum of harmonics. Harmonic *k* has a base amplitude of
1/*k* up to a "knee", and rolls off more steeply above it. The result is then
shaped by a fixed spectral envelope (body formant, presence peak and low-pass).
Harmonic amplitudes follow the voice's pitch as it moves.

| Control | Default | What it does |
|---|---|---|
| `NUM_HARMONICS` | 16 | Harmonics per voice. More harmonics sound brighter and render more slowly. Harmonics above 0.45 × `SAMPLE_RATE` are dropped automatically. |
| `ROLLOFF_LOW` | 1.9 | Rolloff exponent for a voice at or below `ROLLOFF_LOW_HZ`. Higher values sound smoother and rounder. |
| `ROLLOFF_HIGH` | 1.8 | Rolloff exponent for a voice at or above `ROLLOFF_HIGH_HZ`. Lower values sound brighter and more like a sawtooth. |
| `ROLLOFF_LOW_HZ`, `ROLLOFF_HIGH_HZ` | 80, 200 Hz | Pitch range over which the rolloff (and knee) blend from the low setting to the high setting, measured in octaves. The bass stays round and voices brighten as they climb. |
| `KNEE_HIGH` | 4.0 | High voices keep harmonics 1 through this number at full 1/*k* strength before the steeper rolloff starts. Bass voices use a knee of 1. Raising it gives a brighter, buzzier tone. |
| `DIVE_SMOOTH_POWER` | 0.3 | How quickly a diving voice takes on the smoother tone of its bass target. Values below 1 switch early in the dive, which avoids a buzzy sweep down through the mids. |

### Spectral envelope

| Control | Default | What it does |
|---|---|---|
| `FORMANT_FREQ` | 650 Hz | Center of a fixed "body" resonance. |
| `FORMANT_GAIN_START` | 3.0 | Formant boost during the wander. 3.0 means 4× at the center, matching the reference's strong 2nd and 3rd harmonics in the opening. |
| `FORMANT_GAIN` | 0.6 | Formant boost at the peak. The formant eases from the start value to this one during the crescendo, which keeps the chord's mids under control. |
| `FORMANT_WIDTH` | 0.6 oct | Width of the formant (standard deviation, in octaves). |
| `PRESENCE_FREQ` | 3500 Hz | Center of a "presence" resonance that only the harmonics of high voices reach. |
| `PRESENCE_GAIN` | 0.0 | Strength of the presence peak. 0 turns it off. Try 0.5–1.5 for more edge on the soaring treble. |
| `PRESENCE_WIDTH` | 0.6 oct | Width of the presence peak. |
| `LOWPASS_START` | 2500 Hz | Low-pass corner during the quiet wander. It gives a darker, distant sound. |
| `LOWPASS_PEAK` | 12000 Hz | Low-pass corner at the peak. The filter opens with the crescendo, like a bowed string played harder. |
| `LOWPASS_CURVE` | 0.6 | How the filter opens over the crescendo. Values below 1 open early. Values above 1 open late. |

## Phases 3 and 4: Hold and release

| Control | Default | What it does |
|---|---|---|
| `HOLD_TIME` | 5.5 s | How long the full chord holds at peak after the last voice lands. |
| `SAG_DB_PER_S` | 2.0 dB/s | Rate of the gentle decline after the hold, for the bass voices. |
| `SAG_TREBLE` | 0.15 | Additional sag rate per octave above D1, so the treble fades first. For example, A6 (about 5.6 octaves up) sags at roughly 1.8 × the base rate. |
| `SAG_TIME` | 4.5 s | How long the sag lasts before the final fade begins. The sag also continues through the fade. |
| `FADE_OUT` | 3.0 s | Length of the final cos² fade to silence. |
| `TAIL_SILENCE` | 1.0 s | Silence added after the fade. |
| `TARGET_PEAK` | 0.891 | The finished mix is normalized so its loudest sample reaches this level (0.891 = −1 dBFS). |

## Stereo

| Control | Default | What it does |
|---|---|---|
| `STEREO` | True | Renders in stereo. Set it to False for mono. |
| `BASS_CENTER_FREQ` | 120 Hz | Voices landing below this pitch stay near the center. Voices above it are panned by pitch, from low on the left to high on the right. |
| `BASS_SPREAD` | 0.5 | Bass voices alternate left and right by this amount. 0 puts them dead center. 1 pans them hard. |
| `STEREO_OPEN_T` | 10 s | How long the pitch-based panning takes to open from mono to full width. |
| `STEREO_WIDEN` | 0.4 | Extra width from a mid/side boost of the side signal. 0 means normal width. |
| `EARLY_PAN_START` | −0.9 | The opening cluster starts here. −1 is hard left. |
| `EARLY_PAN_PEAK` | 0.55 | It then sweeps right to this position… |
| `EARLY_PAN_LR` | 4.5 s | …taking this long… |
| `EARLY_PAN_RC` | 3.5 s | …and then returns to the center over this long. |

## Output and reproducibility

| Control | Default | Scripts | What it does |
|---|---|---|---|
| `PLAY` | True | play_v2 only | Plays the result through `sounddevice` after rendering. |
| `SAVE_WAV` | True | play_v2 only | Also writes `deep_note.wav`. The other two scripts always write it. |
| `SEED` | 19690105 | all | Random seed. The same seed and settings always produce the same output. Change it for a different random wander, onsets and detune. |
| `OUT_DIR` | script folder | all | Where `deep_note.wav` and `deep_note.png` are written. Every script overwrites the same `deep_note.wav`. |

> **Note:** The random drift curves are drawn on a grid that spans the
> piece's whole length. Any control that changes the total duration (glide
> times, hold, sag, fades) therefore slightly changes the wander pattern too,
> even when `SEED` stays the same.
