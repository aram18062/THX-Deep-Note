# Contributing to THX Deep Note — a Python recreation

Thank you for your interest in this project! Contributions are welcome: bug reports, better documentation, performance improvements and sound-design ideas.

This is a small, single-script project, so the process is lightweight. Please read the guidelines below before opening an issue or a pull request.

## Ways to contribute

- **Report a bug.** Something crashes, a setting is rejected when it shouldn't be, or the output sounds wrong.
- **Suggest a sound-design change.** You think a phase, the timbre or the stereo image could be closer to the original.
- **Improve the code.** Make it faster, clearer, lighter on memory, or more robust, *without changing the sound* (see [Code-only changes](#code-only-changes-must-not-change-the-sound)).
- **Improve the documentation.** Fix typos or unclear explanations, or bring `deep_note_controls.md` up to date with the current defaults.

## Reporting bugs

Please [open an issue](../../issues) that includes:

1. The exact command you ran, including all options (for example, `python deep_note_master.py --no-play --seed 42`).
2. The full output or error message.
3. Your operating system, Python version, and the versions of NumPy and SciPy:

   ```bash
   python -c "import sys, numpy, scipy; print(sys.version, numpy.__version__, scipy.__version__)"
   ```

4. Any constants you changed at the top of `deep_note_master.py`.

Playback problems (no audio device, PortAudio errors) usually come from `sounddevice` rather than this project. If it happens, check whether `--no-play` works and the WAV file sounds right.

## Suggesting sound-design changes

Sound is subjective, so please start with an issue before opening a pull request. Describe:

- **What** you want to change and **why**, for example, "the treble climb arrives too early compared with the reference at ~17 s."
- **Which constants** or which part of `render()` it affects.
- **Evidence**, where possible: the seed you used, a short audio clip, or a spectrogram compared with `original_analysis.png`.

When you experiment, write to a **different output file** so you don't overwrite the reference render:

```bash
python deep_note_master.py --no-play --output my_experiment.wav
```

Keep the seed the same between A/B comparisons, so you compare your change and not a different random take.

## Development setup

1. Fork the repository and clone your fork.
2. Install the dependencies (Python 3.10 or newer):

   ```bash
   python -m pip install numpy scipy sounddevice
   ```

3. Create a branch for your change:

   ```bash
   git checkout -b my-change
   ```

Tip: for faster iteration, temporarily set `SAMPLE_RATE = 48000` while experimenting. Rendering at the default 192 kHz is about four times slower. Remember to set it back before committing.

## Coding guidelines

Please keep to the style of the existing code:

- **Single file.** The synthesizer is a self-contained script. Please don't split it into a package or add new dependencies without discussing it in an issue first.
- **Settings are module-level constants** at the top of the file, grouped by phase, each with a short comment that says what it does, its unit, and which direction to turn it.
- **Validate new settings** in `validate_config()`, with a clear error message.
- **Command-line overrides are local.** New options must be passed into `render()` as arguments and must never modify module globals.
- **Don't disturb the main random stream.** New random effects need their own seeded generator (see `final_wobble_hz_multiplier()` and `harmonic_amps()`). That way, turning an effect on or off doesn't reshuffle the rest of the piece.
- **Keep memory in check.** At 192 kHz a single full-length array is about 48 MB. Delete large arrays once you're done with them (`del`), as the existing code does, and avoid holding one array per harmonic or per voice.
- **Comment on the *why*** where it isn't obvious, especially for anything that affects the sound.

## Pull requests

1. Keep each pull request focused on one change.
2. Update `README.md` (and `deep_note_controls.md`, where relevant) if you add or change a setting or command-line option.
3. **Don't commit rendered audio** (`.wav` files) unless the maintainer asks you to. Mention the seed and settings instead, so others can render it themselves. `deep_note_master.wav` is only updated when the default sound is intentionally changed.
4. Describe what you changed, how you tested it, and whether the default output is still bit-identical.

## Copyright and trademarks

- **Don't add copyrighted audio, scores or text** that you don't have the right to share. This includes other recordings of the THX Deep Note.
- **THX** and **Deep Note** are trademarks of THX Ltd. This project is an independent, non-commercial recreation. Please don't add THX logos or wording that suggests an affiliation or endorsement.
- By submitting a contribution, you agree that it is licensed under the same license as this project (see [`LICENSE`](LICENSE)).

## Questions

If something here is unclear, [open an issue](../../issues) and ask. Thank you for helping to make this recreation better!
