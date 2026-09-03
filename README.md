# [Free Transcription Stack](https://dandjlab-cell.github.io/free-transcription-stack/)

Four free, open-source tools wired in the right order, run on one rented GPU. Tested head to head against Gemini 3.5 Transcribe on the same 103.97 minutes of audio, three trials per configuration.

**[Download the PDF](Free_Transcription_Stack.pdf)** · by [@dandjlab](https://www.instagram.com/dandjlab)

## The stack

| | tool | job |
|---|---|---|
| 01 | **ffmpeg** | clean the audio, mono PCM16 WAV, 16 kHz |
| 02 | **whisper** | the words and the timing, faster-whisper 1.2.1, large-v3 pinned |
| 03 | **pyannote** | who speaks when, pyannote.audio 4.0.7, community-1 |
| 04 | **numpy** | pause, energy, pitch and pacing for every word, measured inside the frozen word boundaries |
| 05 | **modal** | one rented GPU runs it all, warm worker, four clips in parallel |

Every tool is free and open source. The GPU time is not, but it fits inside Modal's $30 monthly credit at this scale.

## What came back

Same 103.97 minutes of real footage. Three repeat trials per configuration.

| metric | free stack | gemini 3.5 transcribe |
|---|---|---|
| median processing time | 5m 57s | 4m 06s to 4m 12s |
| estimated cost per run | $0.319 | $0.520 |
| cost difference | 38.7% cheaper | : |
| monthly cost to you | $0 within Modal's $30 credit | paid API |
| word output | 17,494 every trial | 16,043 to 17,979 |
| repeatability | words and speakers byte-identical | changed between trials |
| zero-duration words | 29 | 516 to 649 |
| timeline reversals | 0 | 29 to 72 |
| overlaps | 0 | 46 to 88 |
| missing section | none found | **9m 34.4s skipped, 3 of 3 times** |
| diarization | speaker identities across the whole set | per clip |
| tone and pacing | included as prosody | not provided |
| transcription API calls | zero | Files + Interactions APIs |

Gemini was about two minutes faster. Its words-only mode skipped the same 9m 34.4s section on all three runs while still reporting `completed`. On that clip it returned 750 to 767 words, against 2,394 from the free stack, which found 1,595 words inside the skipped stretch.

Cost figures use measured runtime and published rates, not invoices. No accuracy claim is made in either direction: blinded human WER/DER scoring is pending.

## Hand this to Claude or Codex

> Build the Free Stack to this spec. Use the exact package versions, model revisions and inference settings below. Do not substitute anything on the never list. When you are done, run the same audio three times and show me the checksums.

### Pinned runtime

| | setting | value |
|---|---|---|
| environment | Debian / Python 3.11 | CUDA 12 / cuDNN 9 |
| audio | ffmpeg | mono PCM16 WAV / 16 kHz |
| package | faster-whisper | 1.2.1 |
| backend | CTranslate2 | 4.8.1 |
| model | Systran/faster-whisper-large-v3 | rev `53ecf83a5bedc5597eb8c8b34eac29e5345520ff` |
| precision | `compute_type=float32` | not fp16, not int8 |
| execution | `WhisperModel(num_workers=4)` | native transcribe, four clips at once |
| inference | `vad_filter=False` | `word_timestamps=True` |
| | `temperature=0.0` | `condition_on_previous_text=False` |
| speakers | pyannote.audio | 4.0.7 |
| model | pyannote/speaker-diarization-community-1 | rev `3533c8cf8e369892e6b79ff1bf80f7b0286a54ee` |
| batching | segmentation=8 / embedding=8 | one pass over the whole batch |
| numeric | torch 2.8.0 / NumPy 2.4.6 | torchaudio 2.8.0 / soundfile |
| access | Hugging Face token | community-1 weights are gated |
| compute | one warm NVIDIA L40S | 16 CPU / 64 GiB / models load once per worker |

### Order of operations

1. **normalize.** ffmpeg to mono 16 kHz. The original file is never rewritten.
2. **transcribe.** one model in memory, four clips at once, settings above.
3. **freeze.** write word, start, end to disk and never edit them again.
4. **assemble.** every clip into one lane, 2 s of silence between, keep the offset map.
5. **diarize.** one speaker pass over that lane, then map turns back to clip time.
6. **assign.** each word takes the speaker with the greatest positive overlap.
7. **measure.** one prosody row per word: pause, duration, energy, voicing, pitch and slope.
8. **confirm.** a human names the speakers.
9. **publish.** hash the output, then write the transcript.

### If the build is right, these are true

1. The same audio, run three times, gives byte-identical word and speaker files.
2. Every word starts before it ends, and no start runs backward.
3. No two words overlap in time.
4. The word count does not change between runs.
5. A person listens to thirty seconds and confirms the transcript matches.

### Never

- **VAD on.** A VAD path omitted real speech, so production runs without it.
- **30-second chunking.** It lost words at the boundary, so whole clips are transcribed.
- **FP16.** It produced timestamp and word-sequence variance across fresh workers.
- **Speaker IDs per clip.** Speaker 1 in clip 3 is not speaker 1 in clip 7.
- **A model naming people.** Diarization hears voices, not identities.
- **Fuzzy text to time.** Use the timestamps you froze.

## Running it

Modal's Starter plan is $0 plus usage and includes $30 of compute credit a month. You need a Modal account, a Hugging Face token with the pyannote community-1 terms accepted (stored as a Modal secret), and a Python 3.11 image with the pins above. Any other NVIDIA CUDA host works too, as long as you recreate the same five properties Modal gave the job: a pinned environment, a model that loads once and stays warm, reserved memory, storage that survives a crash, and a run you can resume.

---

Not affiliated with Google, Modal, SYSTRAN, or pyannote. Source audio is private and not included.
