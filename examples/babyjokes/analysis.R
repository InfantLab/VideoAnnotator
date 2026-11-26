# Do the tools see what the parents saw? The BabyJokes tutorial's analysis, in R.
#
# The same steps as analysis.py: join VideoAnnotator's results table to the
# parents' ratings in the BabyJokes dataset (Zenodo 10.5281/zenodo.23913806,
# CC BY 4.0), see whether the baby's face smiled when the parent said the baby
# laughed (a 2 x 2 table and Cohen's kappa), then ask the study's question:
# does laughing fall off from the first repetition to the third, and does that
# depend on the game or the baby's age? Base R only.
#
# Usage:
#   Rscript analysis.R RESULTS.csv babyjokes-public-ratings.csv \
#     babyjokes-public-clips.csv [by_clip.csv]

# A clip counts as a smile if at least this share of the baby's sampled faces
# smiled. Change it and see how the agreement moves.
SMILE_SHARE <- 0.25
AU12 <- "AU12_Lip_Corner_Puller"
# Tracks seen in under this share of a clip's frames with people can't be the baby.
MIN_TRACK_SHARE <- 0.2

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
  stop("Usage: Rscript analysis.R RESULTS.csv ratings.csv clips.csv [by_clip.csv]")
}

table <- read.csv(args[1], stringsAsFactors = FALSE)
rated <- merge(
  read.csv(args[2], stringsAsFactors = FALSE),
  read.csv(args[3], stringsAsFactors = FALSE)[, c("file", "age_days")],
  by = "file"
)
# `public/X.mp4` in the dataset's CSVs is video `X` in the table.
rated$video <- sub("\\.mp4$", "", sub("^public/", "", rated$file))

# Each clip's baby: person tracking doesn't say who is who, and in these
# games the baby is the smaller person on screen.
people <- table[table$measure == "person", ]
people$area <- people$width * people$height
tracks <- aggregate(area ~ video + track, people, median)
frames <- aggregate(frame ~ video + track, people, function(f) length(unique(f)))
tracks <- merge(tracks, frames)
tracks$share <- tracks$frame / ave(tracks$frame, tracks$video, FUN = max)
tracks <- tracks[tracks$share >= MIN_TRACK_SHARE, ]
# With only one person found it is usually the parent: a baby lying down,
# held, or half hidden in a seat often isn't detected as a person at all.
tracks <- tracks[ave(tracks$area, tracks$video, FUN = length) >= 2, ]
tracks <- tracks[order(tracks$video, tracks$area), ]
baby <- tracks[!duplicated(tracks$video), c("video", "track")]

# Per clip, the share of the baby's faces that smiled, by each measure.
faces <- merge(table, baby)
au12 <- faces[faces$measure == "action_unit_presence" & faces$label == AU12, ]
deepface <- faces[faces$pipeline == "face_analysis" & faces$measure == "emotion", ]
deepface$happy <- deepface$label == "happy"
share <- function(rows, column, name) {
  out <- data.frame(video = character(), x = numeric())
  if (nrow(rows) > 0) out <- aggregate(rows[[column]], list(video = rows$video), mean)
  setNames(out, c("video", name))
}
smiles <- merge(
  share(au12, "value", "au12_share"),
  share(deepface, "happy", "happy_share"),
  all = TRUE
)

# One clip per repetition: the parent rated each repetition once, so every
# take of it carries the same answer and would count it twice.
rated <- rated[order(rated$take), ]
keys <- paste(rated$child_id, rated$session, rated$joke, rated$repetition)
rated <- rated[!duplicated(keys, fromLast = TRUE), ]
by_clip <- merge(rated, smiles, by = "video", all.x = TRUE)
with_face <- !(is.na(by_clip$au12_share) & is.na(by_clip$happy_share))
cat(sprintf(
  "%d repetitions rated by a parent; %d with the baby's face in the results\n",
  nrow(by_clip), sum(with_face)
))

kappa <- function(a, b) {
  observed <- mean(a == b)
  expected <- mean(a) * mean(b) + (1 - mean(a)) * (1 - mean(b))
  if (expected == 1) NA else (observed - expected) / (1 - expected)
}

agreement <- function(measure, name) {
  rows <- by_clip[!is.na(by_clip[[measure]]), ]
  if (nrow(rows) == 0) {
    cat(sprintf("\n%s: not in these results\n", name))
    return(invisible())
  }
  tool <- rows[[measure]] >= SMILE_SHARE
  parent <- rows$parent_laughed == "Yes"
  cat(sprintf("\n%s: smiled in at least %d%% of the baby's faces\n", name, round(100 * SMILE_SHARE)))
  print(table(
    "parent said" = ifelse(parent, "laughed", "didn't"),
    "tool saw" = ifelse(tool, "smiled", "didn't")
  ))
  cat(sprintf(
    "agree on %d%% of %d repetitions; kappa %.2f\n",
    round(100 * mean(tool == parent)), nrow(rows), kappa(tool, parent)
  ))
}

agreement("au12_share", "OpenFace 3, AU12")
agreement("happy_share", "DeepFace, happy")

seen <- by_clip[with_face, ]
seen$parent <- seen$parent_laughed == "Yes"
seen$smiled <- !is.na(seen$au12_share) & seen$au12_share >= SMILE_SHARE
seen$age_months <- seen$age_days / 30.44

cat("\nShare of repetitions the parent said the baby laughed at:\n")
print(round(tapply(seen$parent, list(seen$joke, seen$repetition), mean), 2))
cat("\n... and the tool saw a smile (AU12):\n")
print(round(tapply(seen$smiled, list(seen$joke, seen$repetition), mean), 2))

# The study's question as a model: repetition, game and age, for the
# parents' answers and for the tool's. With many children, a mixed model
# (lme4::glmer with (1 | child_id)) is the better choice; this is base R.
if (length(unique(seen$joke)) > 1 && length(unique(seen$repetition)) > 1 &&
    length(unique(seen$child_id)) > 1) {
  for (outcome in c("parent", "smiled")) {
    if (length(unique(seen[[outcome]])) < 2) {
      cat(sprintf("\n%s is the same for every repetition: nothing to model.\n", outcome))
      next
    }
    cat(sprintf("\n%s ~ repetition + game + age (logistic):\n", outcome))
    fit <- glm(
      reformulate(c("repetition", "joke", "age_months"), outcome),
      family = binomial, data = seen
    )
    print(round(coef(summary(fit)), 3))
  }
} else {
  cat("\n(Too few children, games or repetitions here to fit a model.)\n")
}

if (length(args) >= 4) {
  write.csv(by_clip, args[4], row.names = FALSE)
  cat(sprintf("\nOne row per repetition: %s\n", args[4]))
}
