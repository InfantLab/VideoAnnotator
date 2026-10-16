# Comparable tools (Process step 3)

**Taken**: 2026-10-16. Desk research from each tool's documentation (sources at the end); not yet
checked hands-on or with researchers. Aim: which words and structures researchers already know,
and which ideas fit spec themes 7–13.

## How each organises the work

| Tool | Top-level container | Unit of work | The scheme (what gets coded) | Reuse |
|---|---|---|---|---|
| **ELAN** | none (one `.eaf` per recording; multi-file search across a corpus) | annotation document | tiers, tier types, controlled vocabularies | **templates** (`.etf`): tiers + types + vocabularies to start a new document |
| **BORIS** | **project** file | **observation** (one or two videos) | **ethogram** (behaviours as state or point events), **subjects**, independent variables | the project itself; ethograms shared, imported, modified |
| **Datavyu** | spreadsheet (`.opf`) | the spreadsheet for a session | **columns** (one per coding pass) holding **codes**; **cells** are timed observations | Ruby scripts; copying the column setup |
| **Noldus Observer XT** | **project** | **observation** | **coding scheme**: subjects, behaviours, modifiers (with colours) | **project templates** (`.otb`) with the scheme and variables; **data profiles** select what to analyse |
| **Label Studio** | **project** (dataset + labelling config + people) | **task** | labelling config (XML of tags such as `TimelineLabels`) | config templates |
| **DeepLabCut** | **project** (folder + `config.yaml`) | video | body parts in the config | the project; new videos analysed without adding them |
| **FiftyOne** | **dataset** | sample (a video, with frames) | label fields | saved **views** (filters) over a dataset |

What recurs: researchers in behavioural coding (BORIS, Observer) know **project** and
**observation**; linguists (ELAN) know **template** and **tier**; ML users (Label Studio,
FiftyOne, DeepLabCut) know **project**, **dataset** and **task**. Nobody says "job" or "batch".
"Observation" is the word psychologists coding behaviour are most likely to know for "one video,
coded".

## Ideas that bear on the themes

**Settings and reuse (themes 9, 12; "recipes").** Observer XT's project template and ELAN's
template both answer "set this up once, apply to each new recording" with a named file that can
be shared, which is close to the recipe idea. BORIS keeps the scheme inside the project and
lets ethograms be exported and imported separately. None of them versions the scheme; changes
are edits. Versioning would be ours, and earns its keep because our settings change outputs
(spec 017 provenance), not just the coding form.

**Many synchronised views (themes 10, 13).** ELAN's model is the strongest precedent: a timeline
viewer plus **grid** (a table of one tier's annotations), **subtitle**, **text** and **timeseries**
viewers, in tabs next to the video, shown or hidden individually, and **all synchronised in
time**: navigate in one and the others follow. That is the shape theme 13 asks for (a table next
to the player, click a row to seek) and a reason not to put data in a modal.

**The timeline (theme 10b).** ELAN: one labelled row per tier; tiers sortable by hierarchy,
type or **participant**. Datavyu: one column per coding pass, cells stacked in time. Observer
XT: an event plot against time, synchronised with video, audio and **imported external data**
(numeric signals), and **colours set per modifier in the scheme** so the same thing is the same
colour everywhere. Label Studio: a `TimeSeries` chart under the video for numeric tracks.
Common ground: labelled rows, grouping by person, numeric tracks alongside categorical ones,
stable colours chosen in the scheme rather than hashed.

**Toggling layers (theme 10a).** ELAN shows and hides tiers from the timeline itself (tier list
on the left, right-click menu). FiftyOne's sidebar lists label fields with a checkbox and counts
and filters by value. Both put the toggle where the thing is, rather than in a separate card.

**Summaries and plots (themes 5, 11).** BORIS's **plot events** (one figure per subject over time)
and **time budget** (durations and counts per subject and behaviour), exportable as images, and
its CSV/TSV/XLS/JSON export are what behavioural researchers expect to get out. Our results table
(spec 025) covers the data; a time-budget-style summary per video or dataset is the obvious view
to add.

**Keyboard-first coding (theme 6, researcher notes).** BORIS codes with one key per behaviour;
Datavyu with the number pad for onset/offset; other tools use `[` and `]` for segment start and
end. If researcher notes get a lane, keys to mark a moment or span are the expected interface.

**Collections and selections (themes 5, 10c).** FiftyOne's saved views (a named filter over a
dataset) are a light way to say "these videos" without copying them, close to our datasets as
"a selection from a folder". DeepLabCut analyses new videos without adding them to the project:
structure doesn't have to be declared before use, which fits the "Try it" idea.

## Not covered by any of them

Trying prompts on a model (theme 7) has no precedent in these tools; the nearest analogues are
LLM prompt playgrounds, which are outside this survey. Worth a separate look at how those show
side-by-side outputs and keep history.

## Next

- Try ELAN, BORIS and Datavyu hands-on with one of our demo clips (ELAN export from spec 025 makes
  the round trip possible) and note what feels familiar.
- Ask pilot researchers which of these they have used; their words should outrank ours.

## Sources

- ELAN manual: [tiers and timeline](https://www.mpi.nl/corpus/html/elan_ug/ch03.html),
  [User's guide](https://www.mpi.nl/corpus/html/elan/pt01.html),
  [navigation and synchronised viewers](https://www.mpi.nl/tools/elan/docs/manual/Sec_How_to_navigate_through_a_document.html),
  [manual index](https://www.mpi.nl/tools/elan/docs/manual/index.html);
  Vincent, [ELAN training slides](https://www.umr7023.cnrs.fr/sites/default/files/u%25uid/VINCENT-ELAN.pdf)
- BORIS: [create project](https://www.boris.unito.it/user_guide/create_project),
  [plots](https://www.boris.unito.it/user_guide/plot/),
  [OpenBehavior summary](https://edspace.american.edu/openbehavior/project/boris/),
  Friard & Gamba 2016, [doi:10.1111/2041-210x.12584](https://www.citedrive.com/en/discovery/scpborisscp-a-free-versatile-opensource-eventlogging-software-for-videoaudio-coding-and-live-observations/)
- Datavyu: [Getting started](https://datavyu.org/user-guide/guide/getting-started.html),
  [glossary](https://datavyu.org/user-guide/guide/reference/glossary.html),
  [design the coding spreadsheet](https://datavyu.org/user-guide/best-practices/codes/design.html),
  [configure columns and codes](https://datavyu.org/user-guide/guide/tutorials/configure-datavyu-codes.html)
- Noldus Observer XT 17: [manage projects](https://noldus.com/shared/resources/book/noldus-product-documentation/chapter/the-observer-xt/page/the-observer-xt-17-file-management-manage-projects),
  [visualize data](https://noldus.com/shared/resources/book/noldus-product-documentation/chapter/the-observer-xt/page/the-observer-xt-17-visualize-data-visualize-data),
  [modifier details](https://noldus.com/shared/resources/book/noldus-product-documentation/chapter/the-observer-xt/page/the-observer-xt-17-set-up-your-project-enter-modifier-details)
- Label Studio: [time series with video](https://docs.humansignal.com/guide/ts+video.html),
  [timeline labels tutorial](https://labelstud.io/tutorials/yolo_timeline_labels)
- DeepLabCut: [workflow tutorial](https://fabriziomusacchio.com/teaching/teaching_assessing_animal_behavior/05_deeplabcut),
  [overview guide](https://github.com/paninski-lab/deepgraphpose/blob/main/src/DeepLabCut/docs/UseOverviewGuide.md)
- FiftyOne: [the App](https://beta-docs.voxel51.com/fiftyone_concepts/app),
  [views cheat sheet](https://docs.voxel51.com/cheat_sheets/views_cheat_sheet.html),
  [0.18 sidebar modes](https://voxel51.com/blog/announcing-fiftyone-0-18-with-app-performance-improvements-sidebar-modes-and-custom-attributes)
- Other tools' segment keys: [Potato](https://potatoannotator.readthedocs.io/en/latest/annotation-types/multimedia/video_annotation/),
  [Python Video Annotator](https://pythonvideoannotator.readthedocs.io/en/master/user-docs/shortcuts/)
