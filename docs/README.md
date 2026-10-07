# VideoAnnotator documentation

VideoAnnotator runs computer-vision and audio pipelines over videos of people (faces, bodies,
speech, scenes, and labels from a vision-language model), keeps a record of what made every result,
and shows the results on the video in its viewer. It runs on your own computer: videos never leave
it.

New here? Start with **[Getting started](usage/GETTING_STARTED.md)**.

## Install

- [Installation](installation/INSTALLATION.md): Windows, macOS and Linux, with or without a GPU,
  and the dev container.
- [Docker](deployment/Docker.md): the CPU and GPU images, and Docker Compose.
- [Access tokens for gated models](installation/ENVIRONMENT_SETUP.md): the Hugging Face token
  that speaker diarization needs.
- [Installation troubleshooting](installation/troubleshooting.md)

## Use

- [Getting started](usage/GETTING_STARTED.md): start the server, open the viewer, run a first job.
- [Command-line examples](usage/demo_commands.md)
- [Pipelines](usage/pipeline_specs.md): what each pipeline does and the files it writes.
- [Getting your results](usage/accessing_results.md): downloading and reading the output files.
- [Configuration](usage/configuration.md) and [environment variables](usage/environment_variables.md)
- [Connecting the viewer with an API key](usage/CLIENT_TOKEN_SETUP_GUIDE.md)
- [Scene detection and person tracking](usage/scene_detection.md)
- [Troubleshooting](usage/troubleshooting.md)

## Run it for a group

- [Security overview](security/README.md): [authentication](security/authentication.md),
  [CORS](security/cors.md) and the [production checklist](security/production_checklist.md).
- [Locales in the Docker images](locale.md)

## Contribute

- [Contributing guide](../CONTRIBUTING.md)
- [Development notes](development/README.md): conventions, output formats, the
  [pre-commit hooks](development/PRE_COMMIT_GUIDE.md).
- [Testing overview](testing/testing_overview.md), [testing standards](testing/testing_standards.md)
  and [coverage](testing/coverage_report.md)
- [Roadmap for v1.6.0](development/roadmap_v1.6.0.md), then [v1.7.0](development/roadmap_v1.7.0.md)
  and [v1.7 to v2.0](development/roadmap_v1.7_to_v2.0.md)
- [Changelog](../CHANGELOG.md)

## Reviewing the software

- [Quick start for JOSS reviewers](GETTING_STARTED_REVIEWERS.md)
- [The JOSS paper](joss.md)

Older documents, kept for the record, are in [archive/](archive/README.md). They describe earlier
versions and may be wrong about this one.
