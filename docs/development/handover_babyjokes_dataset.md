# Handover: the BabyJokes dataset on Zenodo and Databrary

Started 2026-10-29. This covers what Caspar does outside the repo, a brief for the desktop agent
that searches his email, and what happens in VideoAnnotator once the records exist.

## Decided (Caspar, 2026-10-29)

- **Name: BabyJokes.** "LookitLaughter" goes.
- **The open sample, 54 clips of 4 babies, goes on Zenodo** as "BabyJokes: open sample" with its
  own DOI. Every row in the metadata has Lookit consent `accepted`, use of media `public`, sharing
  `public` and no withdrawal (checked 2026-10-29). Ethics covers it.
- **The full corpus, about 1,400 clips, goes on Databrary** as "BabyJokes: full corpus". Only
  authorised researchers can see it; Databrary's institutional agreement is the gate. Lookit's
  consent asked parents about Databrary (the `DatabraryShare` column).
- **VideoAnnotator's demo is 5 clips of one baby, one per joke**, kept inside this repo. The
  tutorial walk-through uses all 54, downloaded from Zenodo.
- **InfantLab/babyjokes gets archived.** Its README first points at the new records.

The demo baby is still to choose (Caspar, from the full corpus if none of the 4 open-sample
babies is good enough). Tried and rejected: 2UWdXP, today's demo baby, who never laughed; and
4JDccE, whose image quality and framing are poor (2026-10-30). A good demo baby laughs at some
jokes and not others, is clearly framed with the parent, and has consent for public use.

## Still to decide

1. **Licence for the open sample.** Suggested: CC BY-NC 4.0 (anyone may reuse it with credit, not
   commercially). Check it against the consent wording the desktop agent finds; if parents agreed
   to "public" use without limits, CC BY 4.0 is also possible.
2. **The parents' free-text comments** (`ParentFeedback`, e.g. "She is very jovial and
   smiles/laughs a little at everything"). They're already public on GitHub, but a citable
   release is a good moment to drop them. The prep script leaves them out unless
   `--keep-feedback`.
3. **Creators on the Zenodo record**: who designed the Lookit study and collected the videos, and
   their ORCIDs.

## Brief for the desktop agent

Paste this to the agent that can read Caspar's email:

> Search my email (all years) for what I need to deposit the BabyJokes dataset: videos of parents
> showing jokes to their babies, collected online through Lookit (MIT's platform, now Children
> Helping Science). It may be called BabyJokes, LookitLaughter, Lookit laughter, baby laughter
> online or the Lookit humour study. Report back, with the date and sender of each email you use:
>
> 1. **Ethics approval** for the Lookit study: committee and institution, reference number,
>    approval date, any amendments, and what it says about sharing the videos (publicly, with
>    researchers, on Databrary). Attach or quote the approved wording if there is one.
> 2. **The Lookit study itself**: its title, study ID or link, when data collection ran, and the
>    consent text parents saw, especially the choices about how their videos may be used
>    (private, scientific, public) and the question about Databrary.
> 3. **The full dataset's size**: how many videos and families. My old README says 1,425 videos.
> 4. **What I sent Jerry for his PhD**: when, what (how many videos, which metadata), how it was
>    sent, and any data sharing agreement or conditions attached. Jerry's full name, institution
>    and supervisor.
> 5. **Databrary**: whether I have (or had) a Databrary account or authorised investigator status,
>    which institution sponsored it, and any Databrary volume already made.
> 6. **Zenodo or DOIs**: any earlier deposit or DOI for this dataset or the babyjokes repo.
> 7. **Credits**: who else worked on the study (design, collection, coding), funders and grant
>    numbers to acknowledge, and the Sage Ethical AI hackathon 2023 contacts.
>
> Don't send, forward or reply to anything. If something isn't in my email, say so rather than
> guessing.

## Zenodo: what Caspar does

1. Run the prep script. It downloads the clips and metadata from the old repo, checks consent on
   every row, and writes the upload folder:

   ```bash
   python scripts/prepare_babyjokes_open_sample.py ~/babyjokes_open_sample
   ```

   It gives `videos/` (54 clips), `metadata.csv`, `README.md` (the dataset card) and `SHA256SUMS`;
   79 MB in all.
2. On [zenodo.org](https://zenodo.org), sign in and choose **New upload**. Use the InfantLab
   account that holds VideoAnnotator's DOI, so both sit together.
3. **Reserve a DOI** first (the "Get a DOI now" button). That DOI goes into the babyjokes README and
   VideoAnnotator before anything is published.
4. Upload the files: the 54 clips (one file each, not a zip, so they can be downloaded one at a
   time), `metadata.csv`, `README.md` and `SHA256SUMS`. That's 57 files; Zenodo allows 100.
5. Fill in the record:

   | Field | Value |
   |---|---|
   | Resource type | Dataset |
   | Title | BabyJokes: open sample |
   | Creators | Caspar Addyman (ORCID), and the study team from the email search |
   | Description | The first three paragraphs of the generated `README.md` |
   | Licence | Decided above (suggested CC BY-NC 4.0) |
   | Keywords | infant humour, laughter, parent–infant interaction, Lookit, developmental psychology, video |
   | Related works | *Is supplemented by* `https://github.com/InfantLab/babyjokes`; *Is referenced by* VideoAnnotator, `10.5281/zenodo.16961751` |
   | Funding | From the email search |
   | Version | 1.0 |

6. Publish. Then send the DOI and the record number (the digits in `zenodo.org/records/<number>`)
   back to this repo's agent.

## Databrary: what Caspar does

Databrary only lets in researchers whose institution has signed its access agreement, and a
volume is only shared once its files have release levels. In order:

1. **Authorisation.** If the email search finds no current authorised investigator status,
   register at [databrary.org](https://databrary.org) and ask for authorisation through Caspar's
   current institution. The institution's signatory has to sign Databrary's agreement, which can
   take weeks, so start this first.
2. **Make a volume** "BabyJokes: full corpus". Its overview (title, description, the Zenodo open
   sample as a related link, the ethics reference) can be public even when the videos aren't.
3. **Upload by family.** One Databrary session per family, with the child's age and the clips as
   files. Before uploading, keep only clips where
   `DatabraryShare` is `yes` and `Withdrawal` is `0`.
4. **Release levels.** Set each file to *Authorized users*: any authorised Databrary researcher can
   see it, nobody else can. The 54 open-sample clips may be *Public*, matching their consent.
5. **Share the volume** and send its link back here.

From then on, a researcher who asks for the full set ("available on request") is sent to
Databrary instead of being emailed files.

## What happens in this repo

Before the records exist:

- [x] `scripts/prepare_babyjokes_open_sample.py` (2026-10-29).
- [ ] The 5-clip demo inside the repo, with results from the six default pipelines, replacing
      the single 2UWdXP clip that today is fetched from the old video-annotation-viewer repo.
- [ ] The new babyjokes README (below), ready to push when the DOI is reserved.

Once the Zenodo DOI is reserved:

- [ ] Push the babyjokes README, then archive the repo (Settings → Archive this repository).
- [ ] `videoannotator demo download babyjokes`: fetches the 54 clips and `metadata.csv` from
      Zenodo into a folder and checks `SHA256SUMS`, for the tutorial.
- [ ] The tutorial walk-through on the 54: run the pipelines as a project, export the table, and
      in R join the parents' `HowFunny` and `LaughYesNo` to what the pipelines found.
- [ ] Cite the dataset DOI in VideoAnnotator's README and docs.

## The new babyjokes README (draft)

```markdown
# BabyJokes (archived)

This repository held code from the 2023 Sage Ethical AI hackathon for analysing videos of parents
showing simple jokes to their babies. It is archived and no longer maintained.

## The data

- **BabyJokes: open sample**: 54 clips of 4 babies with parents' ratings, free to download:
  https://doi.org/<ZENODO DOI>
- **BabyJokes: full corpus**: about 1,400 clips, for researchers through Databrary:
  <DATABRARY VOLUME LINK>

The `LookitLaughter.test/` folder here is the same open sample; please cite the Zenodo DOI.

## The tools

Use [VideoAnnotator](https://github.com/InfantLab/VideoAnnotator), which grew out of this project:
its demo and tutorial use the BabyJokes open sample.

## Hackathon

- Write-up: [Quantifying Parent-Child Interactions](https://medium.com/sage-ai/unlocking-parent-child-interactions-advancing-video-understanding-with-multi-modal-llms-c570ab487183)
- London team: https://github.com/chilledgeek/ethical_ai_hackathon_2023
- US team: https://github.com/yutsai84/Ask-Anything
- Motivating video: https://www.youtube.com/watch?v=mt0Um-ZNbj4

MIT licence for the code. The videos are under the licence on the Zenodo record.
```
