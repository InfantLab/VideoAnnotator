# Research: Refresh Model Defaults (spec 030)

**Date**: 2026-10-17 | **Roadmap**: v1.6.0 Phase 5c ([`roadmap_v1.6.0.md`](../../docs/development/roadmap_v1.6.0.md))
**Inputs**: [`dependency_audit_v1.6.0.md`](../../docs/development/dependency_audit_v1.6.0.md) §1, §4, §7, §8;
[`pipeline_review_v1.6.0.md`](../../docs/development/pipeline_review_v1.6.0.md);
[`pipeline_landscape_v1.6.0.md`](../../docs/development/pipeline_landscape_v1.6.0.md); the pipeline
metadata (`src/videoannotator/registry/metadata/*.yaml`) and code on `1.6-dev` (`4f9a1a0`); PyPI,
GitHub and Hugging Face on 2026-10-17.

Figures are quoted from the source named next to them. "Not found" means I looked and could not
verify it. Nothing here was run: no pipeline, no install, no model download. Every "output change"
below is a prediction until the demo-clip check is run.

## 0. Where the defaults stand today

| Pipeline | Default today (code and metadata) | Library (locked) | Weights |
|---|---|---|---|
| `face_analysis` | DeepFace, `detector_backend: retinaface` (since `647549f`, see CHANGELOG "detects faces with RetinaFace"); emotion, age **and** gender on | deepface 0.0.100 (latest 0.0.102, 2026-10-09) | retinaface 119 MB, emotion 6 MB, age 540 MB, gender 540 MB |
| `speech_recognition` | openai-whisper `base` | openai-whisper 20250625 (sdist, built at install) | 140 MB |
| `speaker_diarization` | `pyannote/speaker-diarization-3.1` | pyannote.audio 4.0.7 (spec 015) | 35 MB, gated |
| `person_tracking` | `yolo11n-pose.pt` + ByteTrack | ultralytics 8.3.183 (latest 8.4.179, 2026-10-17) | 6.3 MB |
| `face_openface3_embedding` | OpenFace 3 via `openface-test==0.1.13` | as pinned | 282 MB |

Torch is `>=2.11,<2.12` in every group (audit §8: torchaudio 2.11 is the last release, and
pyannote.audio 4 imports it).

**Two things I found in the code that the roadmap text doesn't reflect:**

1. **The face detector change is already done.** The roadmap's Face item says to replace the
   default Haar cascade (`detector_backend: opencv`). On `1.6-dev` the default is already
   `retinaface`, in code (`face_pipeline.py` default config) and metadata (`deepface.detector_backend:
   retinaface`), with a CHANGELOG entry. What is left of the Face item is age/gender and the
   emotion model.
2. **Turning age and gender off doesn't stop them running.** `_deepface_actions()` returns
   `["emotion", "age", "gender"]` whenever *any* of `detect_emotions`, `detect_age`,
   `detect_gender` is true. The `detect_age`/`detect_gender` flags only filter what is written.
   So setting both false today still downloads 1.08 GB of age/gender weights and runs both models
   on every face. Switching the defaults off has to fix this, or it saves nothing.

Also: `tests/integration/test_output_baseline.py` leaves `face_analysis` out because "DeepFace
detects no faces in the demo clip". That was true with the Haar cascade; the RetinaFace CHANGELOG
entry says that on the sample clips it "now finds parent and infant in every sampled frame". If
that holds for the demo clip, the baseline can include `face_analysis` for the first time (its
current fixture, `demo_clip_face_detections.json`, comes from another take).

---

## 1. Face (`face_analysis`)

### 1a. Detector: already RetinaFace

- **Evaluation**: RetinaFace (Deng et al., CVPR 2020), WIDER FACE hard test set, AP: "achieving AP
  equal to 91.4%" ([arXiv:1905.00641](https://arxiv.org/abs/1905.00641) abstract). DeepFace's
  README says "RetinaFace overperforms among all face detection models in the portfolio"
  ([serengil/deepface](https://github.com/serengil/deepface)).
- **Licence**: RetinaFace MIT (OpenFace 3's LICENSE names it as "RetinaFace (MIT License)").
  DeepFace MIT.
- **Recommendation**: **no change.** Record the reason in metadata (it is already in the
  CHANGELOG) and add the WIDER FACE figure as the benchmark entry for spec 026.

### 1b. Age and gender off by default

- **Evaluation**: DeepFace README: "Age model got ± 4.65 MAE; gender model got 97.44% accuracy,
  96.29% precision and 95.05% recall" (cited there to a tutorial); "age, gender and race /
  ethnicity models were trained on the backbone of VGG-Face with transfer learning"
  ([serengil/deepface](https://github.com/serengil/deepface)). The training set isn't named in the
  README; no infant evaluation found.
- **Why switch**: the pipeline review (accepted 2026-10-01): "Age and gender are meaningless for
  infants" and "drop age/gender now". Roadmap Phase 5c: "the models were trained on adults and say
  nothing useful about infants".
- **Install weight**: with the fix in §0, 1,080 MB less to download on first use (540 + 540 MB,
  `face_analysis.yaml` `weights:`), and two fewer models per face.
- **Output change**: the default COCO file stops carrying `age`/`gender` in each face's
  attributes. Not a schema change (the keys were always conditional on the flags), but anyone
  reading them by default loses them. Turning the flags back on restores today's output.
- **Recommendation**: **switch**, with the `_deepface_actions()` fix.

### 1c. Emotion model

- **Today**: DeepFace's built-in emotion model (7 classes: "angry, fear, neutral, sad, disgust,
  happy and surprise"). DeepFace gives no emotion accuracy figure in its README. The metadata's
  `deepface.emotion_model: VGG-Face` is not passed to DeepFace anywhere (a setting the code doesn't
  read; spec 026 FR-012 covers those).
- **The comparison the roadmap names**: Uwerikowe, Stamate, Addyman, Lachman and Tomlinson (2026),
  *Selecting Optimal Facial Emotion Recognition Models for Naturalistic Caregiver-Child
  Interactions: Addressing Poor Generalizability in Global Majority Data*, TechRxiv,
  doi:10.36227/techrxiv.177162959.96583644/v1 (already in `paper/paper.bib` as `uwerikowe2026`).
  From its abstract (via search; the PDF returns HTTP 403 to automated fetches): five FER
  frameworks, seven pre-trained variants (DeepFace, FER, HSEmotion, OpenFace 3.0, ResEmoteNet;
  trained on FER2013, RAF-DB and AffectNet), on caregiver–child shared-reading video from
  Mokhotlong district, Lesotho; current systems are "fragile under domain shift".
  **Per-model figures and the paper's recommended model: not found** (full text not retrievable
  here). The maintainer is a co-author and has them.
- **Options**: (a) keep DeepFace's emotion model and say plainly it is adult-trained; (b) a model
  the comparison favours, if DeepFace can't host it this is a new backend, not a default change;
  (c) the pipeline review's own decision, "Replace in Phase 5: face detection and expressions from
  OpenFace 3 or a torch-based detector", which removes TensorFlow and unblocks Python 3.14 and
  opencv-python 5 (audit §1 items 2–3).
- **Recommendation**: **maintainer decision** (spec 030 Clarification Q1). Within this spec's
  scope (defaults, not new pipelines), (a) plus 1b is the safe minimum; anything in (b)/(c) is a
  larger change than a default swap.

**Risks**: re-baselining (face_analysis may enter the baseline for the first time); weights
licence of any new emotion model; `stability: stable` in `face_analysis.yaml` though the review
decided "mark experimental" until it is replaced (not done).

---

## 2. Speech (`speech_recognition`): openai-whisper `base` → faster-whisper `large-v3-turbo`

Two changes in one line of the roadmap, which should be checked separately on the demo clip: the
**runtime** (openai-whisper → faster-whisper) and the **model** (`base` → `large-v3-turbo`).

### Evaluation

- **Runtime** (faster-whisper README, [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper)),
  13 minutes of audio, beam size 5:
  - large-v2 on GPU (RTX 3070 Ti 8GB): openai/whisper fp16 **2m23s, 4708MB VRAM**; faster-whisper
    fp16 **1m03s, 4525MB**; faster-whisper int8 **59s, 2926MB**; batched (batch_size=8) fp16
    17s, 6090MB.
  - small on CPU (Core i7-12700K, 8 threads): openai/whisper fp32 **6m58s, 2335MB RAM**;
    faster-whisper fp32 **2m37s, 2257MB**; int8 **1m42s, 1477MB**.
  - No WER comparison between the two runtimes on the same model (the one WER table is
    distil-large-v3 on YT Commons: transformers 14.801, faster-whisper 13.527). Same weights,
    different decoder, so transcripts are expected to be close but **not bit-identical**.
- **Model** ([openai/whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo)
  card): 809M parameters, pruned from large-v3 with "the number of decoding layers have reduced
  from 32 to 4"; "way faster, at the expense of a minor quality degradation"; Open ASR Leaderboard
  "Mean Wer" 7.83. openai-whisper's own table ([openai/whisper](https://github.com/openai/whisper)):
  base 74M params, ~1 GB VRAM, ~7x relative speed; turbo 809M, ~6 GB VRAM, ~8x (relative to large,
  English, A100). **A WER figure for `base` on the same leaderboard: not found** (not verified).
- **No infant or child-directed-speech evaluation found** for either model (landscape §1: "All
  adult-trained; transcribe caregiver speech, not infant sounds").

### Licence

faster-whisper MIT (GitHub licence API); CTranslate2 MIT; turbo weights MIT. faster-whisper maps
`large-v3-turbo`/`turbo` to `mobiuslabsgmbh/faster-whisper-large-v3-turbo` (`faster_whisper/utils.py`),
which Hugging Face now redirects to `dropbox-dash/faster-whisper-large-v3-turbo` (licence `mit`,
last modified 2025-11-05). That is a **third-party CTranslate2 conversion**, not OpenAI's or
SYSTRAN's own upload: record its repository and revision in provenance (spec 017) and consider
pinning the revision.

### Compatibility with our stack

- faster-whisper 1.2.1 (2025-10-31; the repository had commits on 2026-10-16), `Requires-Python
  >=3.9`, pure-Python wheel. Depends on `ctranslate2<5,>=4.0`, `huggingface-hub>=0.21`,
  `tokenizers`, `onnxruntime` (Silero VAD) and `av` (PyAV, bundles FFmpeg). **Does not depend on
  torch**, so it doesn't touch the torch pin either way (the `audio` group keeps torch for
  pyannote).
- CTranslate2 4.8.3 (2026-10-13) has cp313 wheels for Linux x86_64/aarch64, macOS arm64/x86_64 and
  Windows. **openai-whisper's source build at install goes away** (`tool.uv.extra-build-dependencies`).
- GPU: "The latest versions support only CUDA 12 and cuDNN 9"; it needs cuBLAS and cuDNN 9 for
  CUDA 12 available at runtime. torch 2.11+cu126 brings NVIDIA's cuDNN 9 wheels, but whether
  CTranslate2 finds them without `LD_LIBRARY_PATH` is **unverified**: spike item.
- `huggingface-hub>=0.21`: compatible with the `<1.0` cap in `face-openface3`.
- Options we expose map onto faster-whisper's `transcribe()` (beam_size, best_of, patience,
  length_penalty, temperature, word_timestamps, prepend/append punctuations, suppress_tokens), but
  types differ (`suppress_tokens` is a list of ints there, a comma string in our metadata), and
  `fp16`/`use_fp16` becomes `compute_type`. Its result is a generator of `Segment` objects, not
  openai-whisper's dict: `_process_transcription_result` needs an adapter.
- **Translation**: "the `turbo` model is not trained for translation tasks" and "will return the
  original language even if `--task translate` is specified" (openai/whisper README). Our
  metadata offers `task: translate`. With turbo as default, `translate` would silently not
  translate.

### Install weight and hardware

- Weights: turbo `model.bin` is 1,617,884,929 bytes (~1.6 GB, Hugging Face API) vs `base` 140 MB
  today (`speech_recognition.yaml`): about 11x the first-use download.
- VRAM per openai-whisper's table: ~6 GB (turbo) vs ~1 GB (base); int8 lowers it (figures above
  are for large-v2, not turbo). CPU: no turbo CPU timing found; expect it to be far slower than
  `base` on CPU-only laptops. Spike: time it on the demo clip on CPU.

### Output changes

- Format: WebVTT stays (Principle II). Provenance `models[]` changes from `whisper base`
  (`openai-whisper`, sha256 from the download URL) to a Hugging Face repo and revision.
- Content: different words and segment boundaries are certain with a different model; some change
  is possible even with `base` under faster-whisper. `speech_recognition.vtt` is compared exactly
  by the baseline test, so it will fail and must be re-baselined on purpose.
- faster-whisper's VAD is off by default in non-batched `transcribe` (README: "enabled by default
  for batched transcription"). Leaving it off keeps behaviour closest to today.

### Recommendation

**Switch, in two steps checked separately**: runtime first with `base` (measures the runtime's
effect alone), then the model. Keep the old runtime selectable for studies that need bit-identical
transcripts only if the maintainer accepts keeping openai-whisper's source build as an optional
install (Clarification Q3). Route `task: translate` to a model that translates (or refuse it with a
clear message) rather than silently returning the original language.

**Risks**: 11x download; CPU speed; cuDNN discovery; third-party weights repo; translate.

---

## 3. Diarization (`speaker_diarization`): `speaker-diarization-3.1` → `speaker-diarization-community-1`

### Evaluation ([pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) card, "last updated 2025-09")

DER %, "fully automatic, no forgiveness collar, overlapping speech not skipped":

| Dataset | 3.1 | community-1 |
|---|---|---|
| AISHELL-4 | 12.2 | 11.7 |
| AliMeeting (channel 1) | 24.5 | 20.3 |
| AMI (IHM) | 18.8 | 17.0 |
| AMI (SDM) | 22.7 | 19.9 |
| AVA-AVD | 49.7 | 44.6 |
| CALLHOME (part 2) | 28.5 | 26.7 |
| DIHARD 3 (full) | 21.4 | 20.2 |
| Ego4D (dev.) | 51.2 | 46.8 |
| MSDWild | 25.4 | 22.8 |
| RAMC | 22.2 | 20.8 |
| REPERE (phase2) | 7.9 | **8.9** (worse) |
| VoxConverse (v0.3) | 11.2 | 11.2 |

Better on 10 of 12, equal on one, worse on REPERE. The card: "improved speaker assignment and
counting". pyannote's blog ([pyannote.ai/blog/community-1](https://www.pyannote.ai/blog/community-1)):
"significant marked reductions in speaker confusion", segmentation unchanged from 3.1; no figures
there. No child or parent–child audio evaluation found.

### Licence and setup

CC-BY-4.0 ("will always remain freely accessible"); gated (accept conditions on Hugging Face;
Hub API shows `gated: auto`). The repository is self-contained (segmentation, embedding and PLDA,
about 33 MB in all). Today the pipeline needs three licence acceptances (3.1, segmentation-3.0
and community-1, because pyannote.audio 4 loads PLDA from community-1 even for 3.1, see
`speaker_diarization.yaml` `requires_setup`). After the switch, only community-1 is needed, so
**setup gets simpler** for new users. CC-BY-4.0 asks for attribution: add the citation the card
gives to the metadata.

### Compatibility

"Compatible with pyannote.audio 4.x": already installed (spec 015). No torch, Python or install
change. `min_speakers`/`max_speakers` are supported as today. A new
`exclusive_speaker_diarization` output exists (one speaker at a time); not adopting it keeps the
RTTM content comparable to today. Known: pyannote.audio issue #1963, 4.x using ~6x the VRAM of 3.3
(audit §8); worth a measurement on the demo clip.

### Output changes

RTTM format unchanged. Turn boundaries and speaker labels on the demo clip will likely change
(the baseline has 4 turns: three `SPEAKER_00`, one `SPEAKER_01`). Provenance `models[]` changes to
the community-1 repo and revision.

### Sortformer (the spike, audit §8)

[nvidia/diar_streaming_sortformer_4spk-v2.1](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1):
NVIDIA Open Model License; 117M params; "can detect a maximum of 4 speakers; performance degrades
on recordings with 5 and more speakers". DER (30.4 s latency / 1.04 s latency): DIHARD III Eval
full 19.49 / 20.21; AMI Test IHM 15.90 / 16.67; AMI Test SDM 17.80 / 20.57; CALLHOME-part2 full
10.10 / 11.19 (collar 0.25 s for CALLHOME and CH109, 0 s for DIHARD III, AliMeeting, AMI,
NOTSOFAR1). With no collar on both cards, DIHARD III full is 19.49 (Sortformer) vs 20.2
(community-1), AMI IHM 15.90 vs 17.0, AMI SDM 17.80 vs 19.9. The CALLHOME figures aren't
comparable (different collars). These come from two vendors' own cards, not a common evaluation.

Dependencies: NeMo 3.0.0 (Apache-2.0, `torch>=2.6.0`, no torchaudio in its requirements). Its
`asr` extra brings `wandb`, `hydra-core`, `lightning<=2.4.0,>2.2.1` and three
`nv_one_logger_*` packages including `nv_one_logger_training_telemetry`. pyannote.audio 4.0.7
needs `lightning>=2.4`, so both in one environment means exactly lightning 2.4.0. The card installs
NeMo from git `main`, not a release. Telemetry must be vetted against Principle I.

**Recommendation for Sortformer**: **needs the spike**, and is not a v1.6.0 default either way:
switching runtime and library stack (NeMo) is a bigger change than a default swap, and the
torch-cap benefit only arrives if pyannote leaves the `audio` group entirely.

### Recommendation

**Switch to community-1** in v1.6.0: better or equal on 11 of 12 published sets, same library, no
install change, simpler licence setup, open licence. Run the half-day Sortformer spike in parallel
as input to v1.7.0+ (Clarification Q2 asks whether the switch waits for it).

**Risks**: re-baselining the `.rttm`; REPERE regression (broadcast TV; not our domain); VRAM;
pyannote upstream quiet since 2026-06-30 (audit §8).

---

## 4. Person (`person_tracking`): `yolo11n-pose` → `yolo26n-pose`

### Evaluation (Ultralytics docs, COCO Keypoints val2017, 640 px)

| Model | mAP pose 50-95 | mAP pose 50 | CPU ONNX (ms) | T4 TensorRT10 (ms) | params (M) | FLOPs (B) | Source |
|---|---|---|---|---|---|---|---|
| YOLO11n-pose | 50.0 | 81.0 | 52.4 ± 0.5 | 1.7 ± 0.0 | 2.9 | 7.6 | [docs macro at v8.3.183](https://github.com/ultralytics/ultralytics/blob/v8.3.183/docs/en/macros/yolo-pose-perf.md) |
| YOLO26n-pose | 57.2 (e2e) | 83.3 (e2e) | 40.3 ± 0.5 | 1.8 ± 0.0 | 2.9 | 7.6 | [docs.ultralytics.com/tasks/pose](https://docs.ultralytics.com/tasks/pose/) |

YOLO26 page ([docs.ultralytics.com/models/yolo26](https://docs.ultralytics.com/models/yolo26/)):
"released January 2026"; Residual Log-Likelihood Estimation for keypoints, "up to +7.2 AP over
YOLO11 on COCO pose estimation"; DFL removed; NMS-free one-to-one head. The YOLO26 figures are
labelled "(e2e)", the end-to-end (NMS-free) path; whether our `model.track(...)` call uses that
path, and so whether `iou_threshold` still applies, is **unverified**. No infant evaluation of
either model found (the pipeline review: "YOLO-pose isn't in the infant pose benchmarks").

### Licence

AGPL-3.0 for both (and Enterprise). No change from today; the install notice stays.

### Compatibility

- Needs **ultralytics ≥ 8.4.0** (8.4.0, 2026-01-14, "YOLO26 Models Release"); locked 8.3.183,
  `pyproject.toml` floor `>=8.3.0`. So this is a library upgrade *and* a model change: check
  yolo11n-pose on 8.4.x against the baseline first, then switch the model.
- ultralytics 8.4.179 requires `torch>=1.8.0`: no conflict with the torch pin. It adds
  `ultralytics-platform>=0.1.45` (Python ≥ 3.11; "Typed Python SDK for the Ultralytics Platform
  API", depends on httpx) and `polars`, `nvidia-ml-py`. Recheck that analytics stay off
  (constitution Sync Impact Report: "person tracking switched Ultralytics' analytics on... now off
  for the process"; `tests/unit/test_no_telemetry.py`).
- Weights: `yolo26n-pose.pt` 7,878,574 bytes vs `yolo11n-pose.pt` 6,255,593 bytes (GitHub asset
  sizes). Resolved through `resolve_yolo_model()` the same way.
- Keypoints: trained on COCO Keypoints (17), so the COCO-17 layout we write should hold; confirm
  on the demo clip.

### Output changes

COCO format unchanged; boxes, scores, keypoints and possibly track counts change.
`person_tracks.json` is compared exactly by the baseline test and `person_tracking.json` within
tolerance; both will need re-baselining. `person_tracking.yaml` `variant` and `description` name
YOLO11; the citation is YOLO11's.

### Recommendation

**Switch**, after the ultralytics 8.4 upgrade has been checked alone. The pipeline review plans to
replace Ultralytics with RTMPose (licence, not quality) later; this switch doesn't block that.

**Risks**: two-step re-baseline; e2e/NMS behaviour under tracking; new Ultralytics dependencies
and telemetry.

---

## 5. OpenFace 3 (`face_openface3_embedding`): is `openface-test` the maintained distribution?

- CMU's README ([CMU-MultiComp-Lab/OpenFace-3.0](https://github.com/CMU-MultiComp-Lab/OpenFace-3.0))
  still says to install with "pip install openface-test". No other distribution found: PyPI
  `openface` is an unrelated 2023 wrapper of OpenFace 2 (`GuillaumeRochette/openface`, version
  0.0.0); `openface3` and `openface-3` don't exist on PyPI.
- Activity: the CMU repository's last commit was 2025-06-10 ("Update README.md"); no releases;
  open issues since then (e.g. 2025-08-01 "no openface package", 2025-11-11 "Device Argument Not
  Working as Expected", 2026-08-29 "Chin-Region Landmark Offset"). `openface-test`'s last release
  is 0.1.26 (2025-06-11); its PyPI metadata has no author, homepage or licence; 0.1.26 pins
  `Pillow==9.4.0`, `numpy==1.26.4`, `scipy==1.13`, `huggingface_hub==0.21.0` and others exactly.
- Licence: "ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY" (repository
  `LICENSE`).
- **Recommendation**: **no change.** `openface-test` is the distribution CMU names, but neither
  it nor the repository has changed since June 2025, so "maintained" is too strong: say
  "official, dormant since 2025-06" in the metadata and keep `==0.1.13`. Constitution: "No
  proprietary or research-only weights ship as default model identifiers" — it stays opt-in.

---

## 6. VTC 2 licence

- [LAAC-LSCP/VTC](https://github.com/LAAC-LSCP/VTC): no LICENSE file in the repository root (files:
  `.github`, `.gitmodules`, `.python-version`, `CITATION.bib`, `README.md`, `VTC-2`, scripts,
  docs, `pyproject.toml`, `uv.lock`, ...); GitHub reports no licence; pushed 2026-10-01. Weights
  are the git submodule `huggingface.co/coml/VTC-2` (the audit's `coml/VTC-2.0` now redirects);
  its Hub tags carry **no licence**.
- `pyproject.toml` (vtc 2.2.0): `requires-python = ">=3.13"`, depends on `pyannote-audio>=3.4.0`
  and `segma` (git, pinned commit; segma's repository also reports no licence). README: "Windows is
  not supported for the moment"; needs uv, ffmpeg, git-lfs. So the audit §8 note stands: adopting
  VTC keeps pyannote and the torch cap.
- Evaluation (README): average F1 VTC 2.2 **68.3** vs VTC 1.0 **50.9**; per class VTC 2.2 KCHI
  68.5, OCH 58.1, MAL 70.9, FEM 75.6; "Human 2" averages 69.8. No LENA comparison on that page.
- **Recommendation**: not a default change, so outside this spec's requirements. The licence is an
  open action for the maintainer (email LAAC-LSCP); until it is answered, VTC can't ship.

---

## 7. Summary

| Candidate | Recommendation | Key figure (source) | Output changes | Install change |
|---|---|---|---|---|
| Face detector | No change (already RetinaFace) | WIDER FACE hard AP 91.4% (RetinaFace paper) | none | none |
| Face age/gender off | Switch (+ fix actions bug) | none for infants; DeepFace's own age MAE ± 4.65, gender 97.44% (README) | age/gender attributes gone by default | −1.08 GB weights |
| Face emotion | Maintainer decision (Q1) | Uwerikowe et al. 2026: figures not retrievable here | depends | depends |
| Speech runtime | Switch | large-v2 GPU 2m23s → 1m03s; small CPU 6m58s → 2m37s (faster-whisper README) | small transcript differences possible | no sdist build; CTranslate2 + PyAV + onnxruntime |
| Speech model | Switch (turbo), translate handled | Open ASR mean WER 7.83 (turbo card); base figure not found | different transcripts | +1.5 GB weights; ~6 GB VRAM |
| Diarization | Switch to community-1 (Q2: wait for spike?) | DER better on 10/12 sets, e.g. AMI IHM 18.8 → 17.0, DIHARD 3 21.4 → 20.2 (HF card) | different turns/labels | one licence instead of three |
| Sortformer | Spike; not a v1.6.0 default | DIHARD III full 19.49 DER (NVIDIA card) | — | NeMo stack, telemetry to vet |
| Person | Switch, after ultralytics 8.4 alone | COCO-pose mAP50-95 50.0 → 57.2 (Ultralytics docs) | boxes/keypoints/tracks | ultralytics ≥ 8.4 (+ new deps) |
| OpenFace 3 | No change; keep `==0.1.13` | — | none | none |
| VTC 2 | Licence open (maintainer) | F1 68.3 vs 50.9 (VTC README) | — | — |

**Nothing found breaks the torch pin**: faster-whisper doesn't use torch; ultralytics 8.4 needs
`torch>=1.8.0`; community-1 runs on the pyannote.audio 4.0.7 already installed. **No output
format changes** (WebVTT, RTTM, COCO stay); output *content* changes for every switch, so every
switch re-baselines the demo clip on purpose.

---

## Plan decisions (2026-10-17)

Phase 0 of the plan: the open points above, decided for implementation. "Checked at step N" means
the plan's step N verifies it on the demo clip before going on; the decision says what happens
if the check fails.

### Speech

- **S1. Runtime integration.** `faster_whisper.WhisperModel(name_or_repo, device, compute_type)`
  replaces `whisper.load_model`. `WhisperBasePipeline` keeps device selection, the CUDA→CPU
  fallback and ffmpeg audio extraction. Removed with openai-whisper: the transformers (`"/"` in
  the model id) branch, whose transcription was never implemented (`NotImplementedError`), and the
  openai-whisper encoder helpers (`get_whisper_embedding`, `segment_audio`,
  `_pad_or_trim_embedding`), which nothing outside their own tests calls. A model id with `/` now
  means a CTranslate2 repository on Hugging Face, which faster-whisper loads directly.
  *Alternatives*: keep both runtimes (rejected, Q3); transformers' Whisper (heavier, slower, needs
  its own generation code).
- **S2. Compute type.** New technical setting `compute_type`, default `default`: CTranslate2 keeps
  the weights' stored type (float16) and converts where the device can't run it (float32 on CPU and
  on GPUs below compute capability 7.0, such as a GTX 10-series). `use_fp16: false` maps to
  `float32`, so existing configs keep their meaning. The effective type
  (`WhisperModel.model.compute_type`) is recorded in provenance as the quantisation (Principle
  III). *Alternatives*: `int8_float16` (less VRAM, but changes numerics by hardware; available as a
  setting, not the default); `auto` (same hardware dependence).
- **S3. GPU libraries.** CTranslate2 needs cuBLAS 12 and cuDNN 9 at run time. torch 2.11+cu126
  installs NVIDIA's wheels, but CTranslate2 may not find them. Checked at step 5a on Linux: if it
  doesn't, preload them from the `nvidia.cublas`/`nvidia.cudnn` packages before loading the model.
  Either way, `initialize()` transcribes one second of silence, so a missing library fails during
  initialize, where the existing CUDA→CPU fallback (`cuda_markers`) catches it and says so
  (spec edge case), not mid-job.
- **S4. Weights and provenance.** faster-whisper maps size names to repositories
  (`faster_whisper.utils._MODELS`; `large-v3-turbo` → `mobiuslabsgmbh/faster-whisper-large-v3-turbo`,
  which Hugging Face redirects to `dropbox-dash/…`). Weights go to the Hugging Face hub cache, already
  inside the models directory. Provenance: `hub_ref(repo_id)` (repository + snapshot commit) plus
  the compute type, replacing `whisper_ref`. The revision is recorded, not pinned (FR-007 asks for
  recording; pinning would freeze a third-party repository we don't control either way). Which
  repository id names the cache folder after the redirect is checked at step 5b, and the metadata
  `weights:` entry (`cache: huggingface`, ~1,618 MB) uses it.
- **S5. Translate.** `task: translate` with a model whose name contains `turbo` fails in
  `initialize()`: "large-v3-turbo can't translate; set whisper_model to large-v3 or medium for
  translation". Failing beats switching models silently: the output's model would then differ
  from the configured one (FR-015).
- **S6. Settings.** Each existing setting maps to a `transcribe()` argument of the same name; see
  [contracts/speech-settings.md](contracts/speech-settings.md). `suppress_tokens` (a comma string
  in our metadata) is parsed to a list of ints; unset keeps faster-whisper's `[-1]`, the same as
  openai-whisper's `"-1"`. VAD stays off (`vad_filter=False`), as in openai-whisper.
- **S7. First-run message.** The existing `log_model_download` line names the model and its
  download size before loading; on CPU it adds that a smaller model is one setting away
  (`whisper_model: small`). No automatic switch by hardware (spec Assumptions).
- **S8. Old weights.** openai-whisper `.pt` files already in `<models>/whisper/` are left alone and
  still listed on the Models page; nothing deletes a user's files.

### Speaker turns

- **D1.** Default `pyannote/speaker-diarization-community-1`. `requires_setup` lists only its
  licence (SC-007). A config naming 3.1 still needs 3.1's and segmentation-3.0's licences; the
  guide's setting description says so, and the readiness check (which looks at the default) does
  not. `exclusive_speaker_diarization` is not used (spec Assumptions).

### People

- **P1.** ultralytics `>=8.4,<9` in the `person` extra. Step 4a checks `yolo11n-pose` under 8.4
  against step 0 within the baseline tolerance, and the no-telemetry test (8.4 adds
  `ultralytics-platform`). A difference beyond tolerance is investigated before 4b.
- **P2.** YOLO26 is NMS-free end to end. Step 4b runs the demo clip with `iou_threshold` 0.7 and
  0.3; identical outputs mean the setting has no effect with YOLO26, and its description says so
  (FR-020). Otherwise it stays as is.
- **P3.** Default `models/yolo/yolo26n-pose.pt`, resolved by `resolve_yolo_model()` as today;
  Ultralytics downloads it on first use. Provenance stays `weights_ref` (sha256).

### Faces

- **F1.** `_deepface_actions()` returns only the enabled actions. With none enabled it keeps
  `["emotion"]`, because DeepFace's `analyze()` needs one action; the result isn't written, as
  today. The 6 MB emotion model makes this cheap; replacing `analyze()` with `extract_faces()` for
  detection-only runs is a later change.
- **F2.** `detect_age`/`detect_gender` default `false` in code and metadata; `weights:` lists
  only RetinaFace and the expression model (SC-006: −1,080 MB).

### Checks

- **B1. One machine for every comparison.** The committed fixtures were made on an RTX 4060.
  Each step is compared with the previous step's run on the same machine (step 0 first), so GPU
  differences never enter a diff. The fixtures are regenerated once at step 6 on the
  maintainer's machine; if that GPU differs from the RTX 4060, run twice and re-measure the
  tolerances before committing.
- **B2. Reasons in metadata.** A setting may carry `reason` (one sentence) and `reason_source` (a
  link): why its default is what it is. Shown by `pipelines describe` and returned by the API;
  the viewer doesn't show them yet. See [data-model.md](data-model.md).
