"""THX Deep Note — 30-voice additive synthesizer (master version).

Recreation of the THX Deep Note in Python, by Aram Basmadjian, 2026.

SOUND DESIGN / SIGNAL FLOW
--------------------------
1. WANDER:  Thirty gently drifting tones enter between about 0.5 and 5 s.
2. DIVE:    The lead voice begins its descent at 7.5 s. The remaining voices
            converge on their destination pitches from roughly 11–19 s,
            accompanied by a controlled crescendo and rising brightness.
3. HOLD:    A detuned, just-intonation D chord sustains for 5.2 s after the
            last voice arrives. Individual pitches and upper harmonics
            continue to evolve very slightly.
4. RELEASE: Treble-sensitive level sag, cosine-squared fade, then silence. 
            Timing depends slightly on the random seed.

SPATIAL DESIGN
--------------
An early left-to-right global sweep is separate from individual voice
positions. Those voices remain grouped until 8 s, then move smoothly expand
to a full stereo effect between 8 and 12 s. 

COMPATIBILITY / MAINTENANCE
---------------------------
The numerical synthesis expressions and their order are intentionally
retained from various experiments to preserve the 48 kHz float32 WAV output
bit-for-bit for the default seed. The final chord is intentionally voiced 
for exactly 30 oscillators.

The functions below keep the tested musical randomness independent from
pitch drift and harmonic-motion randomness. Disabling those optional effects
therefore does not reshuffle the underlying music. Sound-export settings
are separate from the synthesizer's internal 64-bit computations.

QUICK START
-----------
    python deep_note_master.py
    python deep_note_master.py --no-play
    python deep_note_master.py --no-play --wav-format pcm24
    python deep_note_master.py --no-play --output take_02.wav
    python deep_note_master.py --no-play --no-overwrite
    python deep_note_master.py --no-play --wobble-cents 0
    python deep_note_master.py --no-play --harmonic-motion-db 0
    python deep_note_master.py --no-play --stereo-spread-mode linear

Install dependencies: pip install numpy scipy sounddevice
(`sounddevice` is needed only for live playback; rendering/WAV export work
without an audio output device.) 
"""

import argparse
import os
import sys
import tempfile
import wave

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.io import wavfile

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
SAMPLE_RATE = 192000         # Hz; approved 48 kHz reference; 192000 also supported
NUM_VOICES  = 30             # the signature THX 30-voice cascade (the chord voicing
                             #   below is written for exactly 30 voices)

# --- Phase 1: wander ------------------------------------------------------
START_F_MIN   = 140.0        # Hz - voices begin as random tones in this range (low freq)
START_F_MAX   = 270.0        # Hz - voices begin as random tones in this range (high freq)
WANDER_SEMITONES = 1.5       # depth of each voice's random pitch drift (std, semitones)
WANDER_RATE   = 0.25         # random control points per second (lower = lazier drift)
ONSET_MIN     = 0.5          # s - first voice enters (piece starts in silence)
ONSET_MAX     = 5.0          # s - last voice enters
VOICE_FADE_IN = 1.0          # s - each voice's own fade-in
FADE_IN       = 6.5          # s - global swell up to the plateau level
FADE_IN_POWER = 1.5          # curve of that swell (>1 = slower start)
PLATEAU_DB    = -22.0        # dB - level of the wander section relative to the peak

# --- Phase 2: dive -------------------------------------------------------
GLIDE_START   = 11.0         # s - the bass voices start diving here (before the climb)
GLIDE_STAGGER = 2.5          # s - higher target notes start up to this much later
GLIDE_JITTER  = 0.5          # s - random extra delay per voice
GLIDE_T_MIN   = 3.0          # s - shortest glide/portamento
GLIDE_T_MAX   = 4.5          # s - longest glide/portamento
CROSSED_START = True         # True: the highest voices in the wander become the bass notes and
                             #   the lowest become the top notes, so dive and climb cross
LEAD_DIVE_START = 7.5        # s - one lead voice (heading for D1) starts diving alone here,
                             #   from LEAD_START_HZ, down through the cluster
LEAD_DIVE_T   = 8.5          # s - and takes this long to land
LEAD_DIVE_POWER = 1.8        # its curve: departure (>1 = slow at first, then plunging) ...
LEAD_DIVE_EASE_OUT = 2.0     # ... and arrival (>1 = eases into D1 instead of slamming into it)
LEAD_START_HZ = 300.0        # Hz - where the lead voice starts (just above the 140-270 Hz cluster,
                             #   so it is heard diving down through it); None = bottom of cluster
LEAD_ONSET_DELAY = 1.0       # s - extra delay added to the lead voice's random entry time
                             #   (ONSET_MIN..ONSET_MAX), so it enters later than chance alone
LEAD_GAIN_DB  = 7.0          # dB - the lead voice swells to this much louder than the others ...
LEAD_SWELL_T  = 3.0          # s - ... over this long, arriving exactly as its dive starts; it then
                             #   fades back to normal as it lands, so the chord stays balanced
LEAD_BODY     = 0.6          # body-resonance gain for the lead voice (others: FORMANT_GAIN_START
                             #   during the opening); lower = its fundamental dominates, so it is
                             #   heard as ONE diving note rather than a note plus a diving octave
SOLO_WANDER   = 0.55         # from the start of the lead's swell, the other voices' wander is
                             #   calmed to this fraction (and stays calmed for the rest of the
                             #   piece), so no other voice seems to start diving early
LEAD_WANDER   = 0.7          # semitones (std) - the lead's wander before its dive is scaled to this,
                             #   a typical amount for the cluster, whatever the random draw
CRESC_START   = 11.7         # s - crescendo begins; reaches peak when the last voice lands
UP_THRESHOLD  = 400.0        # Hz - voices landing above this sweep UP (the soaring treble):
UP_GLIDE_START = 12.0        # s - they take off here (+ GLIDE_JITTER) ...
UP_GLIDE_T_MIN = 6.0         # s - ... and climb for between this (shortest climb) ...
UP_GLIDE_T_MAX = 6.5         # s - ... and this (longest climb); each voice picks at random
UP_GLIDE_POWER = 1.3         # ... take-off: 1.0 = leaves at full speed, >1 = gentler start
UP_GLIDE_EASE_OUT = 2.0      # ... arrival: >1 = climb slows as it nears the chord (1.0 = no slowdown)

# --- Final chord ---------------------------------------------------------
D1_FREQ       = 37.8         # Hz - the reference is tuned ~half a semitone sharp (A4 ~ 453.6)
DETUNE_BASS   = 10.0         # cents - voices sharing a note are spread evenly over +/- this
                             #   at the bottom of the chord (so unisons throb every few
                             #   seconds instead of slowly cancelling out) ...
DETUNE_TOP    = 6.0          # ... and over +/- this at the top (F#6 is exposed)
DETUNE_RANDOM = 2.0          # cents (std) - small extra random detune per voice
UPPER_FLAT_CENTS = 12.0      # cents - the reference's upper chord sits this much flat of
                             # pure ratios over D1 (ramps in over the first 3 octaves)
BASS_BOOST_DB = 12.0         # dB - extra gain on the low (bass) voices
BASS_BOOST_FREQ = 90.0       # Hz - crossover for the bass boost (low = favors D1)
BOOST_RAMP_POWER = 3.0       # the boost fades in as glide progresses so a bass-bound voice
                             #   doesn't jump out (as a loud extra voice) when its dive begins
TREBLE_TILT_DB = 2.0         # dB - upper voices settle this much quieter once the LAST voice
                             #   has landed (the shared arrival time, not each voice's own)
TILT_TIME     = 2.0          # s - (over this long), so the climb itself stays at full strength
BASS_CENTER_FREQ = 120.0     # Hz - voices below this sit near the center ...
BASS_SPREAD   = 0.5          # ... alternating left/right by this much (0 = dead center)
NUM_HARMONICS = 16           # harmonics per voice (Dr. Moorer originally used 12)...
ROLLOFF_LOW   = 1.9          # ... harmonic k has base amplitude 1/k**rolloff, where rolloff
ROLLOFF_HIGH  = 1.8          #     goes from ROLLOFF_LOW (smooth, for a voice at <= ROLLOFF_LOW_HZ)
ROLLOFF_LOW_HZ = 80.0        #     to ROLLOFF_HIGH (bright, sawtooth-like, at >= ROLLOFF_HIGH_HZ),
ROLLOFF_HIGH_HZ = 200.0      #     so the bass stays round and voices brighten as they climb ...
BRIGHT_HARMONICS = 4.0       # higher voices keep harmonics 1..BRIGHT_HARMONICS strong (1/k) and roll
                             #   off at their rolloff above that: bright but cello-like, not a
                             #   raw sawtooth (bass voices: n_bright 1, i.e. steep from the start)
DIVE_SMOOTH_POWER = 0.3      # diving voices smooth toward their bass tone as they glide
                             #     (<1 = early in the dive; prevents a buzzy downward sweep) ...
FORMANT_FREQ  = 650.0        # ... shaped by a fixed 'body' resonance centred here (Hz),
FORMANT_GAIN_START = 3.0     #     its boost during the opening (3.0 = 4x at the center: the
                             #     reference's opening voices have very strong 2nd/3rd harmonics)
FORMANT_GAIN  = 0.6          #     ... easing to this at the peak (keeps the chord's mids in check),
FORMANT_WIDTH = 0.6          #     its width (octaves, std),
PRESENCE_FREQ = 3500.0       # ... plus a 'presence' resonance here (Hz) that only the high
PRESENCE_GAIN = 0.25          #     voices reach - their bright, soaring edge - (0 = off),
PRESENCE_WIDTH = 0.6         #     width (octaves, std),
LOWPASS_START = 2500.0       # ... and a rolloff above this (Hz) during the quiet wander,
LOWPASS_PEAK  = 12000.0      #     opening up to this at the peak (brightness rises with the
LOWPASS_CURVE = 0.6          #     (<1 = opens early in the crescendo)
                             #     crescendo, like a bowed string played harder). The formant
                             #     gives high voices a 500-1000 Hz body; bass voices keep a
                             #     strong fundamental.

# --- Organic pitch movement AFTER each voice's glide -------------------------
# Slow, independent detuning of every voice; NOT rhythmic vibrato.
# The 2-cent standard deviation is a conservative first experiment. Try 1 or 3.
FINAL_WOBBLE_CENTS = 2.0      # standard deviation of each voice's slow modulation in cents
FINAL_WOBBLE_RATE = 0.15      # independently randomized spline knots per second
FINAL_WOBBLE_START = 0.75     # glide progress p where modulation begins fading in, 0 <= p < 1

# --- Slowly evolving individual upper harmonics ----------------------------
# 0 dB turns D off exactly; harmonics 2..16 vary gently while the fundamental
# retains its unchanged synthesized amplitude. The modulation is applied on
# harmonic_amps()'s coarse grid, with only ONE full-resolution array alive at
# a time. The random generator is separate from both the musical RNG and C.
HARMONIC_MOTION_DB = 0.7      # per-harmonic spline-knot standard deviation, dB
HARMONIC_MOTION_RATE = 0.20   # random spline control points per second
HARMONIC_MOTION_START = 0.70  # first audible influence at 70% glide completion
# NOTE: Pitch wobble uses its own deterministic RNG stream so the original pitch
# trajectories, onset times, and per-voice detuning never change across A/B runs.

# --- Phases 3-4: hold and release ----------------------------------------
HOLD_TIME     = 5.2          # s - full chord at peak after the last voice lands
SAG_DB_PER_S  = 1.6          # dB/s - gentle decline after the hold (bass voices)
SAG_TREBLE    = 0.06         # extra sag rate per octave above D1 (treble dies first)
SAG_TIME      = 3.8          # s - how long the sag lasts before the final fade
FADE_OUT      = 3.0          # s - cos^2 fade to silence
TAIL_SILENCE  = 1.0          # s - exact digital silence after the fade
TARGET_PEAK   = 0.891        # -1 dB after normalization

# --- Stereo --------------------------------------------------------------
STEREO        = True         # voices panned low->high, left->right
STEREO_OPEN_T = 10.0         # s - legacy linear mode only; retained for A/B tests
STEREO_SPREAD_MODE = "bloom" # "bloom" = approved delayed expansion; "linear" = legacy comparison
STEREO_SPREAD_START = 8.0    # s - beginning of individual-voice stereo expansion
STEREO_SPREAD_END = 12.0     # s - individual voices reach their full stereo positions
STEREO_WIDEN  = 0.4          # extra stereo width via mid/side boost (0 = normal)
EARLY_PAN_START = -0.70      # opening auto-pan: starts here (-1 = hard left) ...
EARLY_PAN_PEAK  = 0.65       # ... swings to here (right) ...
EARLY_PAN_LR  = 4.5          # s - ... over this long ...
EARLY_PAN_RC  = 5.5          # s - ... then returns to center over this long

PLAY          = True         # play the rendered audio through sounddevice
SAVE_WAV      = True         # save a WAV file unless --no-wav is supplied
OVERWRITE     = True         # True: replace existing WAV automatically; False: protect it

SEED          = 19690105     # default seed: render(seed) is reproducible for a given seed
OUT_DIR       = os.path.dirname(os.path.abspath(__file__))  # default WAV location
DEFAULT_OUTPUT = "deep_note_master.wav"                     # default WAV filename
DEFAULT_WAV_FORMAT = "float32"                              # pcm16, pcm24, or float32

# The final chord: just-intonation ratios over D1 and how many voices sit on each note.
CHORD_RATIOS = [1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 40, 48]
#               D1 A1  D2 A2 D3 A3 D4 A4  D5  A5  F#6 A6
CHORD_COUNTS = [3, 1,  4, 1, 4, 3, 3, 2,  2,  2,  2,  3]   # 30 voices
CHORD_NAMES  = "D1 A1 D2 A2 D3 A3 D4 A4 D5 A5 F#6 A6"


def build_final_chord(num_voices):
    """
    The D chord the voices land on, in JUST intonation (small-integer ratios
    over D1): D1, A1, D2, A2, D3, A3, D4, A4, D5, A5, F#6, A6.  Most voices sit
    in the low and middle notes; F#6 gets two voices and A6 three, adding
    shimmer on top of the brightness that comes from the harmonics below.
    Returns a sorted array of Hz.

    The voicing is written for exactly sum(CHORD_COUNTS) == 30 voices; any other
    count raises ValueError instead of silently dropping or missing notes.
    """
    # zip() silently discards extra entries when its inputs have unequal
    # lengths. Reject mistakes here rather than creating an incomplete chord.
    if len(CHORD_RATIOS) != len(CHORD_COUNTS):
        raise ValueError("CHORD_RATIOS and CHORD_COUNTS must have equal lengths")
    if not all(isinstance(c, (int, np.integer)) and c > 0 for c in CHORD_COUNTS):
        raise ValueError("CHORD_COUNTS must contain positive integer voice counts")
    if not all(np.isfinite(r) and r > 0 for r in CHORD_RATIOS):
        raise ValueError("CHORD_RATIOS must contain positive finite frequency ratios")
    expected = sum(CHORD_COUNTS)
    if num_voices != expected:
        raise ValueError(f"NUM_VOICES must be {expected} (the chord voicing is written for "
                         f"exactly {expected} voices); got {num_voices}")
    voices = []
    for r, c in zip(CHORD_RATIOS, CHORD_COUNTS):
        voices.extend([D1_FREQ * r] * c)
    return np.asarray(sorted(voices))


def timbre_gain(f, lowpass, fgain):
    """Spectral envelope: body formant + presence peak, times a 2nd-order rolloff at `lowpass` Hz."""
    formant = (1.0 + fgain * np.exp(-0.5 * (np.log2(f / FORMANT_FREQ) / FORMANT_WIDTH) ** 2)
               + PRESENCE_GAIN * np.exp(-0.5 * (np.log2(f / PRESENCE_FREQ) / PRESENCE_WIDTH) ** 2))
    return formant / (1.0 + (f / lowpass) ** 2)


def harmonic_wave(phase, amps):
    """
    sum_k amps_k * sin(k*phase), via the Chebyshev recurrence (fast).

    `amps` may be any iterable of per-harmonic amplitude arrays (e.g. the
    generator returned by harmonic_amps), so only one harmonic's amplitude
    curve has to exist at a time.
    """
    amps = iter(amps)
    phase = np.mod(phase, 2.0 * np.pi)
    c2 = 2.0 * np.cos(phase)
    s_prev = np.zeros_like(phase)            # sin(0*phase)
    s_cur = np.sin(phase)                    # sin(1*phase)
    del phase
    out = next(amps) * s_cur
    tmp = np.empty_like(out)
    for a in amps:
        # s_next = 2cos(phase)*s_cur - s_prev, computed in place (reusing s_prev's buffer)
        np.multiply(c2, s_cur, out=tmp)
        np.subtract(tmp, s_prev, out=s_prev)
        s_prev, s_cur = s_cur, s_prev
        np.multiply(a, s_cur, out=tmp)
        out += tmp
        del a
    return out


def register(f):
    """0 for a voice at <= ROLLOFF_LOW_HZ, 1 at >= ROLLOFF_HIGH_HZ."""
    return np.clip(np.log2(f / ROLLOFF_LOW_HZ) / np.log2(ROLLOFF_HIGH_HZ / ROLLOFF_LOW_HZ), 0.0, 1.0)


def rolloff_for(f):
    """Harmonic rolloff exponent (above n_bright) for a voice currently at f Hz."""
    return ROLLOFF_LOW + (ROLLOFF_HIGH - ROLLOFF_LOW) * register(f)


def harmonic_amps(t, freq, lowpass, fgain, glide, target, step=64, *,
                  motion_db=0.0, motion_seed=None, voice_index=0, motion_progress=None):
    """
    Per-harmonic amplitude curves (they follow the pitch through the formant).

    A generator: the curves are computed on a coarse grid and each harmonic is
    interpolated to full length only when it is consumed, instead of holding
    NUM_HARMONICS full-length arrays in memory at once.

    Upper harmonics (2+) gain independent slowly varying amplitudes
    on the same coarse grid. motion_db=0 returns the original exact values.
    No separate full-rate harmonic-modulation arrays are created.
    """
    td, fd, lp, fg, gd = t[::step], freq[::step], lowpass[::step], fgain[::step], glide[::step]
    nyq_guard = 0.45 * SAMPLE_RATE                # drop harmonics that would alias
    rolloff = rolloff_for(fd)
    # A voice diving toward a bass note takes on its final (smooth) tone in step with the
    # glide, instead of sweeping a bright harmonic comb down through the mids (= buzz).
    r_end = rolloff_for(target)
    rolloff = np.maximum(rolloff, rolloff + (r_end - rolloff) * gd ** DIVE_SMOOTH_POWER)
    n_bright = 1.0 + (BRIGHT_HARMONICS - 1.0) * register(fd)
    n_bright = np.minimum(n_bright, n_bright + (1.0 - n_bright) * gd ** DIVE_SMOOTH_POWER * (target < fd))

    if motion_db > 0 and NUM_HARMONICS > 1:
        # Independent of the original voice's musical random stream and of
        # the pitch wobble generator; deterministic for a fixed seed.
        if motion_seed is None:
            motion_rng = np.random.default_rng()
        else:
            motion_rng = np.random.default_rng(
                np.random.SeedSequence([int(motion_seed), int(voice_index), 0xDA7A])
            )
        progress = gd if motion_progress is None else motion_progress[::step]
        blend = np.clip((progress - HARMONIC_MOTION_START) /
                        (1.0 - HARMONIC_MOTION_START), 0.0, 1.0)
        blend *= blend * (3.0 - 2.0 * blend)   # smoothstep
        interval = 1.0 / HARMONIC_MOTION_RATE
        knots = np.arange(int(np.ceil(td[-1] / interval)) + 2, dtype=np.float64) * interval

    for k in range(1, NUM_HARMONICS + 1):
        # Preserve the expression and operation order on the disabled path
        # to guarantee a bit-for-bit regression against the earlier synthesis.
        if motion_db == 0 or k == 1:
            yield np.interp(t, td, timbre_gain(k * fd, lp, fg) / k * np.maximum(k / n_bright, 1.0) ** (1.0 - rolloff) * (k * fd < nyq_guard))
        else:
            base = timbre_gain(k * fd, lp, fg) / k * np.maximum(k / n_bright, 1.0) ** (1.0 - rolloff) * (k * fd < nyq_guard)
            knot_db = motion_rng.normal(0.0, motion_db, size=len(knots))
            changing_db = CubicSpline(knots, knot_db, bc_type="natural")(td)
            # Limit rare spline overshoots. 'motion_db' is a stochastic
            # standard deviation, not a hard bound or a peak amplitude.
            np.clip(changing_db, -3.0 * motion_db, 3.0 * motion_db, out=changing_db)
            base *= np.power(10.0, blend * changing_db / 20.0)
            yield np.interp(t, td, base)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def _finite(name, value):
    if not np.isfinite(value):
        raise ValueError(f"{name} must be a finite number; got {value!r}")


def _positive(name, value):
    _finite(name, value)
    if value <= 0:
        raise ValueError(f"{name} must be greater than 0; got {value!r}")


def _non_negative(name, value):
    _finite(name, value)
    if value < 0:
        raise ValueError(f"{name} must be 0 or more; got {value!r}")


def validate_stereo_spread(mode, start, end):
    """Validate stereo expansion timings, including in legacy linear mode."""
    if mode not in ("bloom", "linear"):
        raise ValueError("STEREO_SPREAD_MODE must be 'bloom' or 'linear'")
    _non_negative("STEREO_SPREAD_START", start)
    _positive("STEREO_SPREAD_END", end)
    if start >= end:
        raise ValueError("STEREO_SPREAD_START must be earlier than STEREO_SPREAD_END")


def stereo_spread_envelope(t, mode, start, end):
    """Return a 0–1 voice-separation envelope.

    "bloom": zero through `start`, smoothstep rise to one at `end`,
    constant thereafter. "linear": earlier 0–STEREO_OPEN_T behavior,
    retained verbatim for A/B listening comparisons.
    """
    if mode == "linear":
        return np.clip(t / STEREO_OPEN_T, 0.0, 1.0)
    x = np.clip((t - start) / (end - start), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def validate_config():
    """Check the module-level settings; raises ValueError with a clear message."""
    g = globals()
    validate_stereo_spread(STEREO_SPREAD_MODE, STEREO_SPREAD_START, STEREO_SPREAD_END)
    for key in ("FINAL_WOBBLE_CENTS", "FINAL_WOBBLE_RATE", "FINAL_WOBBLE_START",
                "HARMONIC_MOTION_DB", "HARMONIC_MOTION_RATE", "HARMONIC_MOTION_START"):
        _finite(key, g[key])
    if HARMONIC_MOTION_DB < 0:
        raise ValueError("HARMONIC_MOTION_DB must be 0 or more")
    if HARMONIC_MOTION_RATE <= 0:
        raise ValueError("HARMONIC_MOTION_RATE must be greater than 0")
    if not 0.0 <= HARMONIC_MOTION_START < 1.0:
        raise ValueError("HARMONIC_MOTION_START must be in [0, 1)")
    if FINAL_WOBBLE_CENTS < 0:
        raise ValueError("FINAL_WOBBLE_CENTS must be 0 or more")
    if FINAL_WOBBLE_RATE <= 0:
        raise ValueError("FINAL_WOBBLE_RATE must be greater than 0")
    if not (0.0 <= FINAL_WOBBLE_START < 1.0):
        raise ValueError("FINAL_WOBBLE_START must be in [0, 1)")
    build_final_chord(NUM_VOICES)                      # check chord tables and voice count
    if not isinstance(SAMPLE_RATE, (int, np.integer)) or SAMPLE_RATE <= 0:
        raise ValueError(f"SAMPLE_RATE must be a positive integer; got {SAMPLE_RATE!r}")
    if not isinstance(NUM_HARMONICS, (int, np.integer)) or NUM_HARMONICS < 1:
        raise ValueError(f"NUM_HARMONICS must be an integer >= 1; got {NUM_HARMONICS!r}")
    for name in ("START_F_MIN", "START_F_MAX", "WANDER_RATE", "VOICE_FADE_IN", "FADE_IN",
                 "FADE_IN_POWER", "GLIDE_T_MIN", "GLIDE_T_MAX", "LEAD_DIVE_T", "LEAD_DIVE_POWER",
                 "LEAD_DIVE_EASE_OUT", "LEAD_SWELL_T", "UP_THRESHOLD", "UP_GLIDE_T_MIN",
                 "UP_GLIDE_T_MAX", "UP_GLIDE_POWER", "UP_GLIDE_EASE_OUT", "D1_FREQ",
                 "BASS_BOOST_FREQ", "TILT_TIME", "ROLLOFF_LOW_HZ", "ROLLOFF_HIGH_HZ",
                 "BRIGHT_HARMONICS", "FORMANT_FREQ", "FORMANT_WIDTH", "PRESENCE_FREQ",
                 "PRESENCE_WIDTH", "LOWPASS_START", "LOWPASS_PEAK", "FADE_OUT",
                 "TARGET_PEAK", "STEREO_OPEN_T", "EARLY_PAN_LR", "EARLY_PAN_RC"):
        _positive(name, g[name])
    for name in ("WANDER_SEMITONES", "ONSET_MIN", "ONSET_MAX", "GLIDE_START", "GLIDE_STAGGER",
                 "GLIDE_JITTER", "LEAD_DIVE_START", "LEAD_ONSET_DELAY", "LEAD_WANDER",
                 "CRESC_START", "UP_GLIDE_START", "HOLD_TIME", "SAG_DB_PER_S", "SAG_TREBLE",
                 "SAG_TIME", "TAIL_SILENCE", "LEAD_BODY", "FORMANT_GAIN_START", "FORMANT_GAIN",
                 "PRESENCE_GAIN", "DIVE_SMOOTH_POWER", "BOOST_RAMP_POWER", "SOLO_WANDER",
                 "DETUNE_RANDOM"):
        _non_negative(name, g[name])
    for name in ("PLATEAU_DB", "LEAD_GAIN_DB", "DETUNE_BASS", "DETUNE_TOP", "UPPER_FLAT_CENTS",
                 "BASS_BOOST_DB", "TREBLE_TILT_DB", "ROLLOFF_LOW", "ROLLOFF_HIGH",
                 "LOWPASS_CURVE", "BASS_CENTER_FREQ", "BASS_SPREAD", "STEREO_WIDEN",
                 "EARLY_PAN_START", "EARLY_PAN_PEAK"):
        _finite(name, g[name])
    if LEAD_START_HZ is not None:
        _positive("LEAD_START_HZ", LEAD_START_HZ)
    if START_F_MIN > START_F_MAX:
        raise ValueError("START_F_MIN must not exceed START_F_MAX")
    if ONSET_MIN > ONSET_MAX:
        raise ValueError("ONSET_MIN must not exceed ONSET_MAX")
    if GLIDE_T_MIN > GLIDE_T_MAX:
        raise ValueError("GLIDE_T_MIN must not exceed GLIDE_T_MAX")
    if UP_GLIDE_T_MIN > UP_GLIDE_T_MAX:
        raise ValueError("UP_GLIDE_T_MIN must not exceed UP_GLIDE_T_MAX")
    if ROLLOFF_LOW_HZ >= ROLLOFF_HIGH_HZ:
        raise ValueError("ROLLOFF_LOW_HZ must be below ROLLOFF_HIGH_HZ")
    if LEAD_SWELL_T > LEAD_DIVE_START:
        raise ValueError("LEAD_SWELL_T must not exceed LEAD_DIVE_START (the swell would start "
                         "before the piece does)")
    # Pan values are fractions of a full left/right position. Out-of-range
    # values otherwise get silently clipped in the mixer and mask typos.
    for name in ("EARLY_PAN_START", "EARLY_PAN_PEAK"):
        if not -1.0 <= g[name] <= 1.0:
            raise ValueError(f"{name} must be between -1.0 (left) and +1.0 (right)")
    if not 0.0 <= BASS_SPREAD <= 1.0:
        raise ValueError("BASS_SPREAD must be between 0 and 1")
    if TARGET_PEAK > 1.0:
        raise ValueError(f"TARGET_PEAK must be at most 1.0 (0 dBFS); got {TARGET_PEAK!r}")
    if not isinstance(SEED, (int, np.integer)) or SEED < 0:
        raise ValueError(f"SEED must be a non-negative integer; got {SEED!r}")


# ---------------------------------------------------------------------------
# Independent, slow final-chord pitch modulation
# ---------------------------------------------------------------------------
def final_wobble_hz_multiplier(t, progress, voice_index, seed, cents):
    """Frequency multiplier for an individual voice's organic final-chord drift.

    Generated on an isolated seeded stream: enabling the effect never consumes
    values from the original music RNG. Uses very slow cubic interpolation, a
    3-sigma limit for anomalous overshoot, and a smooth fade-in over the last
    quarter of the voice's frequency glide. Frequencies remain strictly > 0.

    With cents == 0 the caller bypasses this function; disabled
    modulation is exactly transparent to the original oscillator waveform.
    """
    if cents == 0:
        return 1.0
    if seed is None:
        stream = np.random.default_rng()  # fresh take: independent from main RNG
    else:
        stream = np.random.default_rng(
            np.random.SeedSequence([int(seed), int(voice_index), 0xDEE9])
        )
    step = 1.0 / FINAL_WOBBLE_RATE
    knots = np.arange(int(np.ceil(t[-1] / step)) + 2, dtype=np.float64) * step
    deviations = stream.normal(0.0, cents, size=len(knots))
    # Natural boundary conditions make the sparse knots less prone to overshoot.
    modulation = CubicSpline(knots, deviations, bc_type="natural")(t)
    modulation = np.clip(modulation, -3.0 * cents, 3.0 * cents)
    blend = np.clip((progress - FINAL_WOBBLE_START) / (1.0 - FINAL_WOBBLE_START), 0.0, 1.0)
    blend *= blend * (3.0 - 2.0 * blend)  # smoothstep
    return np.exp2(blend * modulation / 1200.0)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
# Sentinel distinguishes the caller not specifying the lead pitch from
# explicitly requesting None (the optional bottom-of-cluster starting pitch).
_USE_CONFIGURED_LEAD_PITCH = object()


def render(seed=SEED, wobble_cents=None, harmonic_motion_db=None,
           stereo_spread_mode=None, spread_start=None, spread_end=None,
           lead_hz=_USE_CONFIGURED_LEAD_PITCH, lead_delay=None):
    """
    Render the piece. Returns (samples, duration_s, n_channels, arrival_s).

    Each call builds its own random generator from `seed`, so the same seed and
    settings always give the same audio. Pass seed=None for a fresh random take.
    Pass wobble_cents=0 to disable final-note pitch movement, or
    harmonic_motion_db=0 to disable upper-harmonic amplitude evolution.
    stereo_spread_mode="linear" selects the legacy voice-spread timeline;
    "bloom" is the approved delayed expansion from 8 to 12 seconds.
    lead_hz and lead_delay are optional per-render overrides. They do not
    modify module globals or affect subsequent render() calls.
    """
    validate_config()
    # Resolve local overrides without mutating global settings; the original
    # defaults and the order of all random draws remain unchanged.
    chosen_lead_hz = LEAD_START_HZ if lead_hz is _USE_CONFIGURED_LEAD_PITCH else lead_hz
    chosen_lead_delay = LEAD_ONSET_DELAY if lead_delay is None else lead_delay
    if chosen_lead_hz is not None:
        _positive("lead_hz", chosen_lead_hz)
    _non_negative("lead_delay", chosen_lead_delay)
    if wobble_cents is None:
        wobble_cents = FINAL_WOBBLE_CENTS
    if harmonic_motion_db is None:
        harmonic_motion_db = HARMONIC_MOTION_DB
    _non_negative("wobble_cents", wobble_cents)
    _non_negative("harmonic_motion_db", harmonic_motion_db)
    mode = STEREO_SPREAD_MODE if stereo_spread_mode is None else stereo_spread_mode
    start = STEREO_SPREAD_START if spread_start is None else spread_start
    end = STEREO_SPREAD_END if spread_end is None else spread_end
    validate_stereo_spread(mode, start, end)
    if seed is not None and (not isinstance(seed, (int, np.integer)) or seed < 0):
        raise ValueError(f"seed must be a non-negative integer or None; got {seed!r}")
    rng = np.random.default_rng(seed)
    final_freqs = build_final_chord(NUM_VOICES)
    octaves = np.log2(final_freqs / final_freqs[0])        # 0 .. ~5.6
    rank = octaves / octaves.max()                          # 0 = bass, 1 = top (30 voices span >1 note)

    # Per-voice randomization ------------------------------------------------
    onsets  = rng.uniform(ONSET_MIN, ONSET_MAX, NUM_VOICES)
    start_f = rng.uniform(START_F_MIN, START_F_MAX, NUM_VOICES)
    if CROSSED_START:              # final_freqs is sorted low->high: give the bass the highest starts
        start_f = np.sort(start_f)[::-1]
    glide_start = GLIDE_START + GLIDE_STAGGER * rank + rng.uniform(0, GLIDE_JITTER, NUM_VOICES)
    glide_t = rng.uniform(GLIDE_T_MIN, GLIDE_T_MAX, NUM_VOICES)
    up = final_freqs > UP_THRESHOLD                      # the soaring upper voices
    glide_start[up] = UP_GLIDE_START + rng.uniform(0, GLIDE_JITTER, up.sum())
    glide_t[up] = rng.uniform(UP_GLIDE_T_MIN, UP_GLIDE_T_MAX, up.sum())
    lead = 0                                             # lowest target (D1) ...
    if chosen_lead_hz is not None:                        # ... starting high, above the cluster
        start_f[lead] = chosen_lead_hz
    elif CROSSED_START:                                  # ... or at the bottom of the cluster
        low = np.argmin(start_f)
        start_f[[lead, low]] = start_f[[low, lead]]
    glide_start[lead], glide_t[lead] = LEAD_DIVE_START, LEAD_DIVE_T
    onsets[lead] += chosen_lead_delay                     # random entry + offset
    if onsets[lead] >= LEAD_DIVE_START:
        print(f"warning: the lead voice enters at {onsets[lead]:.2f} s, after its dive starts "
              f"({LEAD_DIVE_START:.2f} s)", file=sys.stderr)
    elif onsets[lead] + VOICE_FADE_IN > LEAD_DIVE_START:
        print(f"warning: the lead voice enters at {onsets[lead]:.2f} s and is still fading in "
              f"when its dive starts ({LEAD_DIVE_START:.2f} s)", file=sys.stderr)
    detune  = rng.normal(0.0, DETUNE_RANDOM, NUM_VOICES)
    for note in np.unique(final_freqs):                 # spread each unison evenly
        idx = np.where(final_freqs == note)[0]
        if len(idx) > 1:
            half = DETUNE_BASS + (DETUNE_TOP - DETUNE_BASS) * rank[idx[0]]
            detune[idx] += half * np.linspace(-1.0, 1.0, len(idx))
    detune -= UPPER_FLAT_CENTS * np.clip(octaves / 3.0, 0.0, 1.0)
    end_f   = final_freqs * 2.0 ** (detune / 1200.0)
    amp     = 1.0 / NUM_VOICES

    # Bass boost: voices landing on low notes get extra gain (shelves off upward).
    boost_lin = 10.0 ** (BASS_BOOST_DB / 20.0)
    bass_gain = 1.0 + (boost_lin - 1.0) / (1.0 + (final_freqs / BASS_BOOST_FREQ) ** 2)
    chord_tilt = 10.0 ** (-TREBLE_TILT_DB * rank / 20.0)

    # Timeline ---------------------------------------------------------------
    arrival   = float(np.max(glide_start + glide_t))
    # The last arrival depends on the random seed, so check after drawing glides
    # but before allocating large arrays. A zero/negative crescendo window is invalid.
    if CRESC_START >= arrival:
        raise ValueError(
            f"CRESC_START ({CRESC_START:.3f}s) must precede the final chord "
            f"arrival ({arrival:.3f}s) for seed {seed!r}"
        )
    hold_end  = arrival + HOLD_TIME
    fade_start = hold_end + SAG_TIME
    fade_end  = fade_start + FADE_OUT
    duration  = fade_end + TAIL_SILENCE
    n = int(duration * SAMPLE_RATE)
    t = np.arange(n, dtype=np.float64) / SAMPLE_RATE

    # Shared envelope (per-voice sag and entrance are applied in the loop) ---
    swell = np.clip(t / FADE_IN, 0.0, 1.0) ** FADE_IN_POWER
    q = np.clip((t - CRESC_START) / (arrival - CRESC_START), 0.0, 1.0)
    cresc = 10.0 ** (PLATEAU_DB * (1.0 - q) / 20.0)        # dB-linear rise to 0 dB
    u = np.clip((t - fade_start) / FADE_OUT, 0.0, 1.0)
    fade = np.cos(0.5 * np.pi * u) ** 2                     # final fade to silence
    fade[t >= fade_end] = 0.0                               # ... exact silence from fade_end on
    env = swell * cresc * fade
    lowpass = LOWPASS_START * (LOWPASS_PEAK / LOWPASS_START) ** (q ** LOWPASS_CURVE)  # filter opens with the crescendo
    fgain = FORMANT_GAIN_START + (FORMANT_GAIN - FORMANT_GAIN_START) * q          # body eases back
    settle = np.clip((t - arrival) / TILT_TIME, 0.0, 1.0)
    sag_time = np.clip(t - hold_end, 0.0, None)             # seconds into the sag

    # Stereo -----------------------------------------------------------------
    nch = 2 if STEREO else 1
    if STEREO:
        mix = np.zeros((n, 2), dtype=np.float64)
        pans = np.zeros(NUM_VOICES)
        high = np.where(final_freqs >= BASS_CENTER_FREQ)[0]
        if len(high) > 1:
            pans[high] = np.linspace(-1.0, 1.0, len(high))
        low = np.where(final_freqs < BASS_CENTER_FREQ)[0]
        pans[low] = BASS_SPREAD * np.where(np.arange(len(low)) % 2 == 0, -1.0, 1.0)
        width = stereo_spread_envelope(t, mode, start, end)
        pan_lr = EARLY_PAN_START + (EARLY_PAN_PEAK - EARLY_PAN_START) * np.clip(t / EARLY_PAN_LR, 0.0, 1.0)
        pan_rc = EARLY_PAN_PEAK * (1.0 - np.clip((t - EARLY_PAN_LR) / EARLY_PAN_RC, 0.0, 1.0))
        early = np.where(t < EARLY_PAN_LR, pan_lr, pan_rc)
    else:
        mix = np.zeros(n, dtype=np.float64)

    # Coarse time grid for the random pitch drift
    tp = np.arange(int(duration * WANDER_RATE) + 4) / WANDER_RATE

    for i in range(NUM_VOICES):
        # Phase 1: slow random drift in log-frequency around the start tone
        drift = CubicSpline(tp, rng.normal(0.0, 1.0, len(tp)))(t)
        if WANDER_SEMITONES == 0:
            # Disable pitch wander for *all* voices, including the lead.
            # Previously the lead branch divided by WANDER_SEMITONES.
            drift.fill(0.0)
        elif i == lead:                                  # give the lead a typical, audible wander
            pre = t < LEAD_DIVE_START
            pre_std = float(drift[pre].std())
            if pre_std <= np.finfo(np.float64).eps:
                raise ValueError("lead wander has zero variance before LEAD_DIVE_START")
            drift = (drift - drift[pre].mean()) * LEAD_WANDER / (WANDER_SEMITONES * pre_std)
        else:                                            # calm the others during the lead's solo
            t0 = LEAD_DIVE_START - LEAD_SWELL_T
            hold = np.clip((t - t0) / LEAD_SWELL_T, 0.0, 1.0)
            hold = hold * hold * (3.0 - 2.0 * hold)
            k0 = np.searchsorted(t, t0)
            drift = drift[k0] + (drift - drift[k0]) * (1.0 - hold * (1.0 - SOLO_WANDER)) if k0 < len(t) else drift
        f_wander = start_f[i] * 2.0 ** (WANDER_SEMITONES * drift / 12.0)
        del drift

        # Phase 2: glide from the (still drifting) wander pitch to the target: the drift keeps
        # moving during the glide, its influence fading out as p -> 1. Upper voices take off,
        # climb and ease into their notes; the others ease in and out (smoothstep).
        g = np.clip((t - glide_start[i]) / glide_t[i], 0.0, 1.0)
        if up[i]:
            p = 1.0 - (1.0 - g ** UP_GLIDE_POWER) ** UP_GLIDE_EASE_OUT
        elif i == lead:
            p = 1.0 - (1.0 - g ** LEAD_DIVE_POWER) ** LEAD_DIVE_EASE_OUT
        else:
            p = g * g * (3.0 - 2.0 * g)
        freq = f_wander ** (1.0 - p) * end_f[i] ** p
        del f_wander, g

        # Once glide progress reaches 75%, each voice develops an
        # independent, slow final-note drift; it continues through the tail.
        # Apply to frequency BEFORE phase integration to avoid phase jumps.
        # With zero depth, this path is an exact no-op for A/B testing.
        if wobble_cents > 0:
            freq *= final_wobble_hz_multiplier(t, p, i, seed, wobble_cents)

        # Phase = 2*pi * integral(freq dt)
        phase = 2.0 * np.pi * np.cumsum(freq) / SAMPLE_RATE

        # Per-voice entrance and treble-faster sag
        entry = np.clip((t - onsets[i]) / VOICE_FADE_IN, 0.0, 1.0)
        sag_rate = SAG_DB_PER_S * (1.0 + SAG_TREBLE * octaves[i])
        sag = 10.0 ** (-sag_rate * sag_time / 20.0)

        tilt = 1.0 + (chord_tilt[i] - 1.0) * settle
        boost = 1.0 + (bass_gain[i] - 1.0) * p ** BOOST_RAMP_POWER   # boost arrives late in the dive
        if i == lead:                                    # swell up into the solo dive, fade out on landing
            sw = np.clip((t - (LEAD_DIVE_START - LEAD_SWELL_T)) / LEAD_SWELL_T, 0.0, 1.0)
            sw = sw * sw * (3.0 - 2.0 * sw)                  # smooth swell
            boost = boost * (1.0 + (10.0 ** (LEAD_GAIN_DB / 20.0) - 1.0) * sw * (1.0 - p))
        gain = amp * boost * tilt * env * entry * sag
        del boost, tilt, entry, sag
        amps = harmonic_amps(t, freq, lowpass, np.minimum(fgain, LEAD_BODY) if i == lead else fgain,
                             p ** 3 if i == lead else p, end_f[i],
                             motion_db=harmonic_motion_db, motion_seed=seed,
                             voice_index=i, motion_progress=p if harmonic_motion_db > 0 else None)
        del freq, p
        s = harmonic_wave(phase, amps)
        del phase, amps
        s *= gain
        del gain

        if STEREO:
            pi = np.clip(early + pans[i] * width, -1.0, 1.0)
            mix[:, 0] += np.cos((pi + 1.0) * np.pi / 4.0) * s
            mix[:, 1] += np.sin((pi + 1.0) * np.pi / 4.0) * s
        else:
            mix += s

    # Stereo widening (mid/side) ---------------------------------------------
    if STEREO and STEREO_WIDEN != 0.0:
        mid = 0.5 * (mix[:, 0] + mix[:, 1])
        side = 0.5 * (mix[:, 0] - mix[:, 1]) * (1.0 + STEREO_WIDEN)
        mix[:, 0] = mid + side
        mix[:, 1] = mid - side

    if not np.all(np.isfinite(mix)):
        raise RuntimeError("render produced non-finite samples (NaN/inf) - check the settings")
    peak = float(np.max(np.abs(mix)))
    if peak <= 0.0:
        raise RuntimeError("render produced only silence - nothing to normalize")
    mix *= TARGET_PEAK / peak
    return mix, duration, nch, arrival


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def play(samples):
    """Play through sounddevice. Returns True on success; never raises."""
    try:
        import sounddevice as sd
    except (ImportError, OSError):     # OSError: PortAudio missing
        print("sounddevice/PortAudio is not available - skipping playback "
              "(run: pip install sounddevice)", file=sys.stderr)
        return False
    try:
        sd.play(samples.astype(np.float32), SAMPLE_RATE)
        print(f"playing... ({len(samples) / SAMPLE_RATE:.1f}s)")
        sd.wait()
    except KeyboardInterrupt:
        sd.stop()
        print("playback stopped", file=sys.stderr)
        return False
    except Exception as exc:           # no output device, unsupported rate, PortAudio error...
        print(f"playback failed: {exc}", file=sys.stderr)
        return False
    return True


def write_wav(path, samples, nch, wav_format=DEFAULT_WAV_FORMAT, overwrite=False):
    """Save PCM16, PCM24, or float32 WAV without overwriting by accident.

    The 16-bit path preserves the approved prior quantization method. PCM24 is packed as little-endian, signed 3-byte
    samples in frame/channel order. Float32 uses scipy.io.wavfile to write
    the correct IEEE-float WAV header (format tag 3).

    Raises FileExistsError if destination exists and overwrite=False.
    Failed writes leave no partial new file; when overwriting, the existing
    output is retained until the replacement has been fully written.
    """
    if wav_format not in ("pcm16", "pcm24", "float32"):
        raise ValueError(f"unsupported WAV format: {wav_format!r}")
    if nch not in (1, 2):
        raise ValueError(f"expected mono or stereo audio; got {nch} channels")
    if samples.ndim != (1 if nch == 1 else 2) or (nch == 2 and samples.shape[1] != 2):
        raise ValueError("sample array shape does not match the channel count")
    if not np.all(np.isfinite(samples)):
        raise ValueError("cannot write WAV containing NaN or infinite samples")

    def write_stream(output):
        if wav_format == "float32":
            # The `wave` module writes PCM only; SciPy correctly writes IEEE
            # float WAV (compression code 3, not PCM code 1).
            # Float retains headroom and is not restricted to [-1, +1].
            wavfile.write(output, SAMPLE_RATE, samples.astype("<f4"))
        else:
            bits = 16 if wav_format == "pcm16" else 24
            with wave.open(output, "wb") as w:
                w.setnchannels(nch)
                w.setsampwidth(bits // 8)
                w.setframerate(SAMPLE_RATE)
                if bits == 16:
                    pcm = np.round(np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
                    w.writeframes(pcm.tobytes())  # round-to-nearest, little-endian PCM
                else:
                    # Chunk to keep peak memory controlled at 192 kHz stereo.
                    # The least significant 3 bytes of little-endian int32
                    # are exactly a signed, two's-complement 24-bit PCM sample.
                    for offset in range(0, len(samples), 65536):
                        frame_block = samples[offset:offset + 65536]
                        pcm32 = np.round(np.clip(frame_block, -1.0, 1.0) * 8388607.0).astype("<i4")
                        packed = pcm32.view(np.uint8).reshape(-1, 4)[:, :3].tobytes()
                        w.writeframesraw(packed)

    # Exclusive open prevents accidental replacement even if a different
    # process creates the destination between checking and writing. For
    # --overwrite, first render into a temporary file in the same directory,
    # then atomically replace the old file only if the write succeeded.
    created = False
    temporary_path = None
    try:
        if overwrite:
            directory = os.path.dirname(os.path.abspath(path))
            with tempfile.NamedTemporaryFile(mode="wb", prefix=".deep_note_", suffix=".wav.tmp",
                                             dir=directory, delete=False) as output:
                temporary_path = output.name
                write_stream(output)
            os.replace(temporary_path, path)
            temporary_path = None
        else:
            with open(path, "xb") as output:
                created = True
                write_stream(output)
    except Exception:
        if temporary_path is not None:
            try:
                os.remove(temporary_path)
            except OSError:
                pass
        if created:
            try:
                os.remove(path)
            except OSError:
                pass
        raise

    tag = "stereo" if nch == 2 else "mono"
    fmt_label = {"pcm16": "16-bit PCM", "pcm24": "24-bit PCM", "float32": "32-bit float"}[wav_format]
    print(f"wrote {path}  ({len(samples) / SAMPLE_RATE:.1f}s, "
          f"{SAMPLE_RATE} Hz, {fmt_label} {tag})")


def non_negative_seed(value):
    """argparse type that rejects invalid/negative seeds with a friendly error."""
    try:
        seed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seed must be a non-negative integer") from exc
    if seed < 0:
        raise argparse.ArgumentTypeError("seed must be a non-negative integer")
    return seed


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Render (and optionally play) the THX Deep Note.")
    ap.add_argument("--lead-delay", type=float, default=None, metavar="SEC",
                    help=f"extra seconds added to the lead voice's random entry time "
                         f"(default: LEAD_ONSET_DELAY = {LEAD_ONSET_DELAY})")
    ap.add_argument("--lead-hz", type=float, default=None, metavar="HZ",
                    help=f"starting pitch of the lead diving voice in Hz "
                         f"(default: LEAD_START_HZ = {LEAD_START_HZ}; the cluster is "
                         f"{START_F_MIN:.0f}-{START_F_MAX:.0f} Hz)")
    ap.add_argument("--seed", type=non_negative_seed, default=SEED,
                    help=f"random seed (default: {SEED}); the same seed produces the same audio")
    ap.add_argument("--output", metavar="PATH", default=os.path.join(OUT_DIR, DEFAULT_OUTPUT),
                    help=f"output WAV filename (default: {DEFAULT_OUTPUT}, next to this script)")
    overwrite_group = ap.add_mutually_exclusive_group()
    overwrite_group.add_argument("--overwrite", action="store_true", dest="overwrite",
                                 help="replace an existing WAV (overrides OVERWRITE setting)")
    overwrite_group.add_argument("--no-overwrite", action="store_false", dest="overwrite",
                                 help="prevent replacement (overrides OVERWRITE setting)")
    ap.set_defaults(overwrite=OVERWRITE)
    ap.add_argument("--wobble-cents", type=float, default=None, metavar="CENTS",
                    help=f"final-note pitch modulation strength in cents; "
                         f"0 disables it (default: {FINAL_WOBBLE_CENTS})")
    ap.add_argument("--harmonic-motion-db", type=float, default=None, metavar="DB",
                    help=f"individual upper-harmonic motion in dB; "
                         f"0 disables it (default: {HARMONIC_MOTION_DB})")
    ap.add_argument("--stereo-spread-mode", choices=("bloom", "linear"), default=None,
                    help=f"stereo spread timing; linear is the previous control "
                         f"(default: {STEREO_SPREAD_MODE})")
    ap.add_argument("--spread-start", type=float, default=None, metavar="SECONDS",
                    help=f"when the delayed stereo bloom begins (default: {STEREO_SPREAD_START})")
    ap.add_argument("--spread-end", type=float, default=None, metavar="SECONDS",
                    help=f"when the stereo bloom reaches full width (default: {STEREO_SPREAD_END})")
    ap.add_argument("--wav-format", choices=("pcm16", "pcm24", "float32"),
                    default=DEFAULT_WAV_FORMAT,
                    help="WAV encoding: pcm16, pcm24, or float32 (default)")
    ap.add_argument("--no-play", action="store_true", help="render without playing")
    ap.add_argument("--no-wav", action="store_true", help="render without writing a WAV")
    return ap.parse_args(argv)


def main(argv=None):
    """Run command-line render and export; do not alter module-level settings."""
    args = parse_args(argv)
    if args.lead_hz is not None:
        if not np.isfinite(args.lead_hz) or args.lead_hz <= 0:
            raise SystemExit("--lead-hz must be a finite positive frequency in Hz")
    if args.lead_delay is not None:
        if not np.isfinite(args.lead_delay) or args.lead_delay < 0:
            raise SystemExit("--lead-delay must be a finite number of seconds, 0 or more")
    try:
        validate_config()
    except ValueError as exc:
        raise SystemExit(f"invalid settings: {exc}") from exc

    saving = SAVE_WAV and not args.no_wav
    if saving and not os.path.isdir(os.path.dirname(os.path.abspath(args.output))):
        raise SystemExit(f"output directory does not exist: "
                         f"{os.path.dirname(os.path.abspath(args.output))}")
    if saving and not args.overwrite and os.path.exists(args.output):
        # Check early, before the expensive render. The exclusive-open check
        # in write_wav also protects against a concurrent create.
        raise SystemExit(f"output already exists: {args.output} "
                         "(choose another --output, enable OVERWRITE, or add --overwrite)")

    selected_lead_delay = LEAD_ONSET_DELAY if args.lead_delay is None else args.lead_delay
    selected_lead_hz = LEAD_START_HZ if args.lead_hz is None else args.lead_hz
    if ONSET_MAX + selected_lead_delay >= LEAD_DIVE_START:
        print(f"warning: with a {selected_lead_delay:.1f} s delay the lead voice may enter after its "
              f"dive starts ({LEAD_DIVE_START:.1f} s), depending on the seed", file=sys.stderr)
    print(f"lead voice starts at {selected_lead_hz:.1f} Hz" if selected_lead_hz is not None
          else "lead voice starts at the bottom of the cluster")
    try:
        samples, duration, nch, arrival = render(
            args.seed, wobble_cents=args.wobble_cents,
            harmonic_motion_db=args.harmonic_motion_db,
            stereo_spread_mode=args.stereo_spread_mode,
            spread_start=args.spread_start, spread_end=args.spread_end,
            lead_hz=selected_lead_hz, lead_delay=selected_lead_delay)
    except (ValueError, RuntimeError) as exc:
        raise SystemExit(f"render failed: {exc}") from exc

    # Save first, so a playback problem can never cost you the render.
    if saving:
        try:
            write_wav(args.output, samples, nch, args.wav_format, overwrite=args.overwrite)
        except (OSError, ValueError) as exc:
            print(f"ERROR: could not save WAV to {args.output!r}: {exc}", file=sys.stderr)
            return 1    # No misleading 'done.' after a failed save.
    if PLAY and not args.no_play:
        play(samples)
    chord = build_final_chord(NUM_VOICES)
    print(f"start: {NUM_VOICES} drifting tones {START_F_MIN:.0f}-{START_F_MAX:.0f} Hz; "
          f"dive from {GLIDE_START:.1f}s, chord lands {arrival:.1f}s "
          f"(D chord {chord.min():.1f}-{chord.max():.1f} Hz: {CHORD_NAMES}); "
          f"total {duration:.1f}s, {'stereo' if nch == 2 else 'mono'}")
    used_wobble = FINAL_WOBBLE_CENTS if args.wobble_cents is None else args.wobble_cents
    print(f"Final-note pitch variation: {used_wobble:g} cents "
          f"({'disabled' if used_wobble == 0 else 'enabled'})")
    used_motion = HARMONIC_MOTION_DB if args.harmonic_motion_db is None else args.harmonic_motion_db
    print(f"Independent harmonic variation: {used_motion:g} dB "
          f"({'disabled' if used_motion == 0 else 'enabled'})")
    spread_mode = STEREO_SPREAD_MODE if args.stereo_spread_mode is None else args.stereo_spread_mode
    spread_a = STEREO_SPREAD_START if args.spread_start is None else args.spread_start
    spread_b = STEREO_SPREAD_END if args.spread_end is None else args.spread_end
    print(f"Stereo spread: {spread_mode}" +
          (f" ({spread_a:.1f}s to {spread_b:.1f}s)" if spread_mode == "bloom" else
           f" (legacy linear, 0s to {STEREO_OPEN_T:g}s)"))
    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
