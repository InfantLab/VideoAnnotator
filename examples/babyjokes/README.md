# BabyJokes: do the tools see what the parents saw?

The analysis for the BabyJokes tutorial, in Python (`analysis.py`, pandas only) and in R
(`analysis.R`, base R only). Both do the same steps and print the same tables.

You need:

- the results table exported from your BabyJokes project in the viewer (Export → Results table,
  CSV), from a run with person tracking, face analysis (DeepFace) and OpenFace 3;
- `babyjokes-public-ratings.csv` and `babyjokes-public-clips.csv` from the dataset,
  [10.5281/zenodo.23913806](https://doi.org/10.5281/zenodo.23913806) (CC BY 4.0).

```bash
python analysis.py BabyJokes.csv babyjokes-public-ratings.csv babyjokes-public-clips.csv --out by_clip.csv
Rscript analysis.R BabyJokes.csv babyjokes-public-ratings.csv babyjokes-public-clips.csv by_clip.csv
```

What they do:

1. **Find the baby.** Person tracking follows each person but doesn't say who is who; in these
   games the baby is the smaller person on screen. Faces share their person's `track`, so the
   baby's faces are the rows on the baby's track. A clip with only one person found gets no baby:
   that person is usually the parent, as a baby lying down, held or half hidden in a seat is often
   not detected at all. Check a few clips on the Video page; on 12 checked, the rule picked the
   baby in 9.
2. **Did the baby smile?** For each clip, the share of the baby's faces with OpenFace 3's AU12
   (lip corner puller) present, and the share DeepFace calls happy. A clip counts as a smile when
   that share is at least `SMILE_SHARE` (0.25; change it and watch the agreement move).
3. **One clip per repetition.** Parents rated each repetition once, so every take of it carries
   the same answer; the last take is kept.
4. **Agreement with the parents**: a 2 × 2 table and Cohen's kappa, for each measure.
5. **The study's question**: the share of repetitions with a laugh (the parents') or a smile (the
   tools'), by game and repetition, and by age; in R, a logistic model of each. With all 25
   children a mixed model (`lme4::glmer(... + (1 | child_id))`) is the better choice.

`tests/unit/test_babyjokes_analysis.py` runs both on the five demo clips bundled in the viewer.
