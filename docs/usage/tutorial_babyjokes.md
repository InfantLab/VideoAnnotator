# Tutorial: do the tools see what the parents saw?

One complete piece of research with VideoAnnotator, on real infant data: get the videos, run
pipelines on them, look at the results, export them, and analyse them in R or Python.

The data is **BabyJokes**: 390 short webcam clips of babies at home while a parent plays five
simple games (Peekaboo, Nom Nom Nom, That's Not a Hat, Tearing Paper, That's Not a Cat), each three
times. After every attempt the parent said whether the baby laughed. That answer is our ground
truth: we didn't make it, and the tools never see it.

The question: **when the parent said the baby laughed, did the tools see the baby smile?** And
then the study's own question: does laughing fall off from the first time a game is played to the
third, and does that depend on the game or the baby's age?

No pipeline detects laughter as such. What we have are faces: OpenFace 3 says, for each face it
finds, how likely each of eight facial movements is, including **AU12**, the lip corners pulled up
in a smile; DeepFace says which expression each face most likely shows. Part of the lesson is how
far these get on home webcam video of infants. (Not far: see [What to take away](#9-what-to-take-away).)

You'll need about two hours of your own time, plus the time the computer takes to run (step 4).

## 0. What you'll end with

- the 390 clips in a project on your VideoAnnotator, with person tracking, face and OpenFace 3
  results for each;
- one results table (CSV) of everything the tools found;
- a 2 × 2 table of the tools against the parents, with Cohen's kappa, and laughing by game,
  repetition and age, from a script in R or Python you can change.

## 1. Install VideoAnnotator and open the demo

Install and start VideoAnnotator as in [Getting started](GETTING_STARTED.md#recommended-start-videoannotator).
When it asks which folder your videos are in, choose (or make) a folder for this tutorial, for
example `Studies` in your home folder.

The viewer opens in your browser. On **Home**, under **Try it on a demo clip**, open **BabyJokes**:
five clips of one family, one per game, with results already made. Nothing is downloaded. This is
what your own results will look like after step 4; [The Video page](video_page.md) explains what's
on screen.

## 2. Get the data

The dataset is on Zenodo: Addyman, *BabyJokes*, version 1.1,
[10.5281/zenodo.23913806](https://doi.org/10.5281/zenodo.23913806) (CC BY 4.0). Download three
files from that page:

| File | What it is |
|---|---|
| `babyjokes-public-v1.0.zip` | the 390 clips (447 MB) |
| `babyjokes-public-clips.csv` | one row per clip: child, session, game, repetition, take, age in days, checksum |
| `babyjokes-public-ratings.csv` | one row per clip: the parent's answers, "Did your baby laugh?" and "How funny?" |

Unzip the clips into the folder you shared in step 1, so you have `Studies/babyjokes/public/`
with 390 `.mp4` files in it. Keep the two CSVs somewhere you'll find them for step 8.

A few things about the data that matter later:

- Clip names say what each clip is: `FVM73L.sess1.joke1.rep2.take1.Peekaboo.mp4` is child FVM73L,
  session 1, the first game played, its second repetition, first take, Peekaboo.
- 22 clips are a second or later take of a repetition. The parent answered once per repetition,
  so every take of it carries the same answer: we keep the last take.
- 25 children, aged about 4 to 29 months; one took part twice.

## 3. Videos and a project

In the viewer, **Videos → Folders** shows the folder you shared. Open `babyjokes/public`: 390
videos, none run yet (– in every column).

Make a project of them so the study stays together: select all the rows, **Add to project ▾ →
New project…**, and call it **BabyJokes**. A project made from a whole folder follows it, so
anything added to the folder later joins it. See [Videos and projects](videos_and_projects.md).

## 4. Run the pipelines

Open the **BabyJokes** project and choose **Run…**. In **What to find**, choose:

- **Person tracking**: finds each person and follows them through the clip. We need it to tell
  the baby from the parent.
- **Face analysis (DeepFace)**: each face's most likely expression.
- **OpenFace 3**: each face's action units, including AU12.

Faces are linked to the person they belong to, so a face's rows share that person's `track`.
Person tracking doesn't say who is the parent and who is the baby; that is a decision for the
study, and we make it in step 8.

You can add the other default pipelines (scene detection, speaker diarization, speech
recognition) if you're curious; the analysis doesn't use them, and they more than double the
time. On **Start**, check the time estimate and **Start run on 390 videos**. You can close the tab:
the run carries on.

How long it takes, measured:

| | Three pipelines above | All six defaults |
|---|---|---|
| Laptop with an NVIDIA RTX 4060 (8 GB), two clips at a time | 1 h 55 min | 3 h 34 min |

Without a GPU it takes much longer: run one game instead (in **Videos**, type `Peekaboo` in the
filter, select the 78 rows and **Run…**) and do step 8 on those.

## 5. Look at one family

When the run has finished, open the **BabyJokes** project and click a clip from FVM73L, the demo's
family. On the Video page:

- the **People** track has a row per person; the baby is usually the smaller box;
- the **Faces** and **Action units** tracks show what OpenFace 3 found, face by face; switch
  overlays on in **Layers** to see the boxes and landmarks on the video;
- **‹ ›** step through the project's clips in order, so you can watch the same game's three
  repetitions one after another.

Compare a clip the parent said the baby laughed at (Peekaboo, repetition 2) with one they said
the baby didn't (That's Not a Hat, repetition 1). Can you see it in the AU12 row? Add a note
(**N**) where you see a laugh: notes go into the results table too.

## 6. Ask AI (optional)

Instead of a facial measure, you can ask a vision model the parent's question directly. In **Ask
AI**, choose a clip and ask "Did the baby laugh?" with a **Yes / No** answer. Try it on a few clips
whose answer you know from the ratings; if it does well, run the question on the project. A model
running on your own computer sends nothing anywhere; [Ask AI](ask_ai.md) explains models, servers
and what is sent where. Its answers come out in the same results table, so step 8's scripts can
use them in place of AU12.

## 7. Results out

Open the **BabyJokes** project, **Export ▾ → Results table (CSV)**, with **Latest results**. For
390 clips it takes about a minute, and the file is about 100 MB: one row per thing found, with the
same columns for every pipeline. [Results out](exporting_results.md#the-table) describes them; the
ones we use:

| Column | For us |
|---|---|
| `video` | the clip's name, without `.mp4` |
| `pipeline` | `person_tracking`, `face_analysis`, `face_openface3_embedding` |
| `track` | which person a person or face row belongs to |
| `measure`, `label`, `value` | for example `action_unit_presence`, `AU12_Lip_Corner_Puller`, `1` |
| `frame`, `x`, `y`, `width`, `height` | where and when |

**Methods**, in the same menu, writes a methods paragraph for a paper: each pipeline, its model
and settings, and what to cite.

## 8. Analysis in R and in Python

The scripts are in the VideoAnnotator repository, in
[`examples/babyjokes/`](https://github.com/InfantLab/VideoAnnotator/tree/master/examples/babyjokes):
`analysis.R` (base R only) and `analysis.py` (pandas only). They do the same steps and print the
same tables. Download one and run it with the table and the two CSVs:

```bash
Rscript analysis.R BabyJokes.csv babyjokes-public-ratings.csv babyjokes-public-clips.csv by_clip.csv
python analysis.py BabyJokes.csv babyjokes-public-ratings.csv babyjokes-public-clips.csv --out by_clip.csv
```

What they do, and the decisions in them, each one a setting at the top you can change:

1. **Find the baby.** In each clip, of the people tracked in at least a fifth of the frames, the
   baby is the smaller. A clip where only one person was found gets no baby: that person is
   usually the parent, as a baby lying down, held, or half hidden in a seat is often not detected
   at all. We checked this rule on 12 random clips: it picked the baby in 9, and in the 3 misses
   the baby hadn't been detected. Check a few yourself on the Video page.
2. **Did the baby smile?** For each clip, the share of the baby's faces with AU12 present
   (OpenFace 3), and the share DeepFace calls happy. A clip counts as a smile when that share is
   at least `SMILE_SHARE`, 0.25.
3. **One clip per repetition**: the last take.
4. **Agreement**: a 2 × 2 table of the parent's answer against the tool's, and Cohen's kappa.
5. **The study's question**: laughing (the parents') and smiling (the tool's) by game and
   repetition, and by age; in R, a logistic model of each.

What we got on all 390 (VideoAnnotator 1.7.0):

```text
368 repetitions rated by a parent; 322 with the baby's face in the results

OpenFace 3, AU12: smiled in at least 25% of the baby's faces
tool saw     didn't  smiled
parent said
didn't          170      35
laughed          77      37
agree on 65% of 319 repetitions; kappa 0.17

DeepFace, happy: smiled in at least 25% of the baby's faces
agree on 53% of 314 repetitions; kappa 0.04
```

And the share of repetitions where the parent said the baby laughed, and where the tool saw a
smile (the 322 repetitions with the baby's face):

| Game | Parent: 1st time | 2nd | 3rd | Tool (AU12): 1st | 2nd | 3rd |
|---|---|---|---|---|---|---|
| Peekaboo | 0.50 | 0.59 | 0.56 | 0.44 | 0.27 | 0.20 |
| That's Not a Cat | 0.50 | 0.40 | 0.36 | 0.21 | 0.35 | 0.23 |
| That's Not a Hat | 0.45 | 0.27 | 0.14 | 0.14 | 0.23 | 0.14 |
| Nom Nom Nom | 0.40 | 0.33 | 0.36 | 0.10 | 0.24 | 0.27 |
| Tearing Paper | 0.09 | 0.14 | 0.29 | 0.14 | 0.19 | 0.29 |

By age, parents said babies under 6 months laughed at 17% of repetitions, and at 38–42% from 6
months on (the 18 repetitions at 12–17 months, from few children, are an exception).

Some things to try: change `SMILE_SHARE` and watch kappa move (it stays between about 0.06 and
0.19); look at the clips where the parent and the tool disagree, listed in `by_clip.csv`; with 25
children, a mixed model with a random effect per child (`lme4::glmer(... + (1 | child_id))`) is
the better model than the plain logistic one.

## 9. What to take away

- **The tools agreed with the parents only slightly.** AU12 agreed on 65% of repetitions, kappa
  0.17: better than chance, not good enough to stand in for the parent. DeepFace's "happy" was at
  chance. Choosing the baby wrongly isn't the reason: leaving out the clips where the baby
  couldn't be found didn't change the agreement.
- **Why.** Both face models were trained on adult faces, mostly facing the camera in good light;
  these are infants, often side-on, small in the frame, in dim rooms, on a laptop webcam. A laugh is
  also more than a smile: a sound, a moment. A parent's answer covers the whole attempt; a share of
  smiling faces is a summary of sampled frames.
- **The parents' answers still answer the study's question.** Laughing depends on the game
  (Peekaboo most, Tearing Paper least) more than on repetition, and babies under 6 months laughed
  least. The tool's smiles are rarer than the parents' laughs and show little of the difference
  between games.
- **Check before you trust.** Every number here rests on choices: which person is the baby, what
  counts as a smile, which take to keep. The Video page is where you check them, clip by clip.

## Cite

If you use the data, cite the dataset: Addyman, C. *BabyJokes*, version 1.1, Zenodo,
[10.5281/zenodo.23913806](https://doi.org/10.5281/zenodo.23913806). For VideoAnnotator and the
models, use the methods paragraph from step 7, which lists what to cite for each pipeline you ran.
