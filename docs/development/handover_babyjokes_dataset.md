# Handover: the BabyJokes demo and tutorial

Started 2026-10-29; rewritten 2026-11-11 once the dataset was published. The October plan (a
54-clip open sample, a brief for an email search, how to deposit on Zenodo and Databrary) is done
or overtaken; it is in git history.

## The records

| | |
|---|---|
| Zenodo, cite this | **BabyJokes**, version 1.1 (2026-11-03): [10.5281/zenodo.23913806](https://doi.org/10.5281/zenodo.23913806) |
| Zenodo, all versions | [10.5281/zenodo.23808041](https://doi.org/10.5281/zenodo.23808041) (resolves to the newest) |
| Zenodo, version 1.0 | 10.5281/zenodo.23808042 (2026-10-30). Has no ratings: don't cite it |
| Databrary | [volume 2013](https://nyu.databrary.org/volume/2013): 1,380 clips, 90 sessions, authorised researchers only |
| Licence | CC BY 4.0 |
| Ethics | Goldsmiths, Department of Psychology, reference 1408 |

Version 1.1's files, each downloadable on its own from
`https://zenodo.org/records/23913806/files/<name>?download=1`:

- `babyjokes-public-v1.0.zip`: 390 `.mp4` clips in `public/`, 447 MB (unchanged from 1.0).
- `babyjokes-public-clips.csv`: one row per clip: `file, child_id, session, joke_position,
  repetition, take, joke, age_days, bytes, sha256`.
- `babyjokes-public-ratings.csv`: one row per clip, the same keys plus `parent_laughed` (Yes/No)
  and `parent_how_funny` / `parent_how_funny_score` (1 Not funny to 5 Extremely funny).
- `README.md`: the dataset card.

What the data looks like (checked 2026-11-11):

- 390 clips, 25 children, 26 sessions (BQKXCD took part twice), aged 111 to 890 days.
- Clips per child run from 9 to 30: 22 clips are re-takes. "Every child has 15 clips" is the wrong
  check; one clip per session × game × repetition is the right one, and only
  `2UWdXP.sess1.joke3.rep1.take1.ThatsNotAHat` is missing (the parent rated that attempt, but
  Lookit saved no video of it; the old babyjokes repo lacks it too).
- Parents said the baby laughed on 155 clips, not on 235. Nobody answered "Extremely funny": in
  practice the scale is 1–4 (143, 142, 88, 17).
- A parent rated each repetition once, so every take of a repetition carries the same rating. Count
  repetitions, not clips (keep the last take), or some ratings count twice.
- The 54 ratings in the old babyjokes sheet match version 1.1 exactly.

Lab only, never in the demo or tutorial: `complete\` and `manifest.csv` in Caspar's
`LookitLaughterRELEASE` folder.

## The demo: 5 clips of one family, in the repo

Done 2026-11-25. **FVM73L** (298 days), chosen by Caspar from AETYX2, DAVN6G and FVM73L, the three
with the best framing: the only one of them with all five games (AETYX2 has no Nom Nom Nom; DAVN6G
never laughed), and the most varied answers. One repetition per game (the parent's answer):

| Clip | Laughed? |
|---|---|
| `FVM73L.sess1.joke1.rep2.take1.Peekaboo` | yes (at every repetition) |
| `FVM73L.sess1.joke2.rep2.take1.ThatsNotACat` | yes (not at rep 1) |
| `FVM73L.sess1.joke3.rep1.take1.ThatsNotAHat` | no (at no repetition) |
| `FVM73L.sess1.joke4.rep1.take1.NomNomNom` | no (at no repetition) |
| `FVM73L.sess1.joke5.rep3.take1.TearingPaper` | yes (rep 3 only, rated 3: its highest) |

- The clips and the six default pipelines' results are in `viewer/public/demo/babyjokes/` (12 MB,
  with a README citing the dataset), so they ship inside the viewer and open offline, on any build;
  `viewer/src/lib/demos.ts` lists them, Peekaboo first. Home names the dataset and its licence.
  They replace `peekaboo-rep3-v1.1.1`, fetched from the old video-annotation-viewer repo.
- H.264, so every browser plays them (the old demo was H.265).
- Made with `videoannotator process` after the speech fix (UAT #76), so Nom Nom Nom's transcript
  reads "Oh, no, no, no, no, no, no, no, no, […]" rather than "no" ×240.
- `tests/integration/test_output_baseline.py` still runs on the old 2UWdXP clip
  (`viewer/demo-assets/`), which is only a test fixture now; it didn't need re-baselining.
- To remake: run the six pipelines on the five clips, copy the files the README lists, and
  `bash scripts/build_viewer.sh`. `viewer/src/test/lib/demos.test.ts` checks every file is there
  and read as what it is.

## The tutorial: the full public set

### The question it answers

**Do the tools see what the parents saw?** For each clip the parent said whether the baby laughed.
The tutorial runs VideoAnnotator over all 390 clips, then checks its findings against the parents'
answers, and asks the study's own question: does laughing fall off from the first repetition to the
third, and does that depend on the game and the baby's age?

That gives a new user one complete research loop (data in, pipelines, look, ask, export, analyse)
on real infant data with a ground truth they didn't make. It is also honest about limits: no
pipeline detects laughter as such. The signals are the face pipeline's expressions, the audio
pipeline's voice emotion, and Ask AI's "Did the baby laugh?" asked of a vision model, and part of
the lesson is how far each gets.

Speech transcripts on these clips are a lesson of their own (chapter 9): Whisper loops on
sing-song games and laughter, Qwen3-ASR doesn't, and on Tearing Paper rep 3, where the parent said
the baby laughed, Qwen3-ASR writes "Hahaha" ([speech_models_v1.7.0.md](speech_models_v1.7.0.md)).

### Chapters

Each chapter is one recording of 3 to 6 minutes and one section of a written page with the same
steps and screenshots. Short chapters can be re-recorded alone when the viewer changes; the
roadmap's 10-minute screencast is cut from them.

| # | Chapter | On screen | Learner's time |
|---|---|---|---|
| 0 | What we'll do | The dataset, the question, what you'll end with | – |
| 1 | Install, open the demo | Start-up program, Docker; the 5-clip demo opens with results, nothing downloaded | 15 min |
| 2 | Get the data | The DOI page; download the zip and both CSVs; unzip into the folder shared with VideoAnnotator | 10 min (447 MB) |
| 3 | Videos and a project | Videos → the shared folder → 390 videos; a project "BabyJokes" from the folder | 5 min |
| 4 | Run the pipelines | New run on the project: which pipelines and why; the time estimate; start it and leave it | overnight (see below) |
| 5 | Look at one family | The Video page: layers, a lane, notes; previous/next through the family's 15 clips; a clip the parent said was funny next to one they didn't | 10 min |
| 6 | Ask AI | "Did the baby laugh?" as a Yes/No answer, on a few clips, then on the project; local model, what's sent where | 20 min + run |
| 7 | Results out | Export the results table (CSV); what its columns are; the methods text | 5 min |
| 8 | Analysis in R and in Python | Join the table to `babyjokes-public-ratings.csv` on the file name; agreement with the parents (a 2 × 2 table, kappa); laughing by repetition × game, by age | 30 min |
| 9 | What to take away | Where the tools agreed with the parents and where not; why (infants, home webcams); cite the tool and the data | – |

### Following along

- **Running all 390 takes time.** The recording starts the run, cuts, and comes back to it
  finished. The learner chooses one of three:
  1. run all 390 overnight (a GPU helps a lot);
  2. run one game, e.g. Peekaboo's 78 clips, and do chapter 8 on those;
  3. skip the wait with our results for all 390 (see below), and still run a few clips themselves
     in chapter 4.
- **Times are measured, not guessed**: the full run on Caspar's laptop GPU (RTX 4060) and a CPU-only
  run of one game, before anything is recorded. The tutorial quotes both.
- **Downloading is by browser from the DOI**, as anyone would. No extra command to learn; the
  checksum check is three lines in chapter 8's scripts, not a step on screen.
- **The analysis scripts live in the repo** (`examples/babyjokes/analysis.R`, `analysis.py`) and a
  test runs them on the demo's results table, so a viewer change that breaks the export breaks CI,
  not the tutorial.
- **Recording**: the release-candidate build, a fresh server with no earlier runs, one fixed window
  size, the viewer in light mode; nothing on screen from Caspar's own data.

### The full run (2026-11-11)

The six default pipelines (face analysis, OpenFace 3, person tracking, scene detection, speaker
diarization, speech recognition; not Ask AI) on all 390, as one run in a project "BabyJokes",
on the dev container: RTX 4060 laptop GPU (8 GB), two clips at a time (`MAX_CONCURRENT_JOBS=2`).

- Download of the zip from Zenodo: 1 min 55 s; all 390 checksums match.
- **390 of 390 finished, none failed, in 3 h 34 min.** 85 min of video (clips 11 s median, 62 s
  longest): 1.8 clips a minute, 2.5 s of wall time per second of video.
- Per clip, start to finish: 58 s median, 97 s for the slowest tenth, 210 s at most.
- Where the pipelines' time goes (seconds per second of video, median):

  | Pipeline | Share | s per s of video |
  |---|---|---|
  | Speaker diarization | 39% | 0.71 |
  | Face analysis | 26% | 0.67 |
  | OpenFace 3 | 14% | 0.38 |
  | Speech recognition | 13% | 0.36 |
  | Scene detection | 6% | 0.17 |
  | Person tracking | 3% | 0.06 |

- 42% of the time is outside the pipelines (UAT #72), so the run could be much shorter.
- The run's results table: 2 min 49 s to export, 91 MB, 295,184 rows (UAT #74).
- Still to measure: a CPU-only run of one game, for the tutorial's other estimate.

### To check before writing it

- Which face is the baby's. Face results hold both parent and baby; the analysis needs the baby's.
  Fixed 2026-11-11 (UAT #75): faces now share their person's `track`, and nothing is labelled
  parent or infant by default, so the tutorial picks the baby's track itself (the smaller person).
  The 390 need the face pipelines and person tracking run again for this.
- How a learner loads our results for option 3: a folder they drop into the shared folder and the
  server picks up, or an import. If neither exists yet, decide whether it's worth building.

## In this repo

- [x] Run the six default pipelines on all 390 on this machine; record times; look at the results
      table at that size (2026-11-11; UAT #72–75).
- [x] Caspar picks the demo family; bundle the 5 clips and their results; switch
      `viewer/src/lib/demos.ts` to them (FVM73L, 2026-11-25; the baseline test keeps its clip).
- [ ] Retire `scripts/prepare_babyjokes_open_sample.py` (it built the 54-clip open sample, which was
      never deposited).
- [ ] The two checks above (whose face; loading our results).
- [ ] `examples/babyjokes/` analysis scripts and their test.
- [ ] The written tutorial in `docs/usage/`, then the recordings.
- [ ] Cite the dataset (10.5281/zenodo.23913806) in the README and the tutorial (the demo's files
      and Home cite it, 2026-11-25).
- [ ] Archive InfantLab/babyjokes, its README pointing at the Zenodo and Databrary records.

## Still to decide

1. **Publishing our results for all 390** (option 3 above): a separate Zenodo record ("BabyJokes:
   VideoAnnotator results", with the VideoAnnotator version that made them, linked to the dataset)
   or a file beside the tutorial. Either way without face embeddings, which are close to biometric.
2. **Which vision model Ask AI uses in chapter 6**: one small enough for a laptop (via Ollama), and
   whether the tutorial also shows a remote model server.
