# Troubleshooting

Possible solutions to commonly encountered problems getting started with this project. If you can't find an answer to your problem here or in online searches, then please [raise an issue](https://github.com/InfantLab/babyjokes/issues)

## Setting up your environment

In general - we use a lot of different large libraries with complex dependencies (torch, tensorflow, opencv, etc). So there some challenges getting your environment working. We always recommend creating a new environment and installing requirements from the `requirements.txt` or the `enviroment.yml` file.

### Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized

This solution seems to help.
<https://stackoverflow.com/questions/20554074/sklearn-omp-error-15-initializing-libiomp5md-dll-but-found-mk2iomp5md-dll-a>

## Code Organization

The codebase is organized as follows:

### Directory Structure
