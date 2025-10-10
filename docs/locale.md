Why we set locale in Docker images

Some base images don't include compiled locales. When the environment sets
LANG or LC\_\* to a UTF-8 locale that isn't present, programs (git, Python,
gettext-enabled programs) print warnings like:

    locale: Cannot set LC_CTYPE to default locale: No such file or directory

This repository standardizes on en_US.UTF-8 for development images. The
project Dockerfiles now generate the locale and export LANG/LC_ALL so
runtime processes see a working UTF-8 locale.

If you prefer a smaller footprint in a restricted image, C.UTF-8 is a
valid alternative (if provided by your base image). In that case, set
ENV LANG=C.UTF-8 and ENV LC_ALL=C.UTF-8 in the Dockerfile.

Why this is safe

- en_US.UTF-8 is the conventional UTF-8 locale used in many CI/dev
  environments.
- Generating the locale only runs during image build and avoids noisy
  warnings at runtime.

If you still see locale warnings after rebuilding, run `locale` in the
container and ensure LANG/LC_ALL are set to a generated locale.
