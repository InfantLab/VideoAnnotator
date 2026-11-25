# Demo clips: BabyJokes, one family

Five clips of one family (child FVM73L, 298 days old), one per game, from the public BabyJokes
dataset: Addyman et al., *BabyJokes*, version 1.1, Zenodo,
[10.5281/zenodo.23913806](https://doi.org/10.5281/zenodo.23913806). CC BY 4.0. Ethics: Goldsmiths,
Department of Psychology, reference 1408.

| Clip | Did the baby laugh? (the parent's answer) |
|---|---|
| `FVM73L.sess1.joke1.rep2.take1.Peekaboo` | yes |
| `FVM73L.sess1.joke2.rep2.take1.ThatsNotACat` | yes (not at the first repetition) |
| `FVM73L.sess1.joke3.rep1.take1.ThatsNotAHat` | no |
| `FVM73L.sess1.joke4.rep1.take1.NomNomNom` | no |
| `FVM73L.sess1.joke5.rep3.take1.TearingPaper` | yes (the third repetition only) |

The results are VideoAnnotator 1.7.0.dev0's six default pipelines (face analysis, OpenFace 3,
person tracking, scene detection, speaker diarization, speech recognition), made 2026-11-25 with
`videoannotator process`; each file records what made it. OpenFace 3's files were made again the same day, after its outputs were found to be misread (UAT #78). OpenFace 3's detailed file and person
tracking's per-person summary are left out to keep the viewer small. The viewer's demos are listed
in `viewer/src/lib/demos.ts`.
