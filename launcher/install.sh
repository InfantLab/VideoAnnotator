#!/bin/sh
# shellcheck shell=sh
#
# Installs videoannotator-start, and a "Start VideoAnnotator" shortcut:
#
#   curl -LsSf https://github.com/InfantLab/VideoAnnotator/releases/latest/download/install.sh | sh
#
# Options: --no-shortcut (just the command), --quiet. Copies the launcher into
# ~/.local/bin; nothing else is installed (spec 024, research R16).

set -eu

VA_GITHUB="InfantLab/VideoAnnotator"
# Replaced with the release version by CI. `videoannotator-start update` sets
# VA_RELEASE to the release it is updating to.
VA_RELEASE=${VA_RELEASE:-@VERSION@}
BIN=${VA_INSTALL_DIR:-$HOME/.local/bin}

shortcut=1 quiet=0
for arg in "$@"; do
    case "$arg" in
        --no-shortcut) shortcut=0 ;;
        --shortcut) shortcut=1 ;;
        --quiet) quiet=1 ;;
        *) echo "Unknown option: $arg" >&2; exit 1 ;;
    esac
done

say() { [ "$quiet" = 1 ] || printf '%s\n' "$*"; }

case "$VA_RELEASE" in
    @VERSION@) url="https://github.com/$VA_GITHUB/releases/latest/download/videoannotator-start" ;;
    v*) url="https://github.com/$VA_GITHUB/releases/download/$VA_RELEASE/videoannotator-start" ;;
    *) url="https://github.com/$VA_GITHUB/releases/download/v$VA_RELEASE/videoannotator-start" ;;
esac

download() {
    if command -v curl >/dev/null 2>&1; then curl -fsSL "$1" -o "$2"
    else wget -qO "$2" "$1"; fi
}

mkdir -p "$BIN"
if ! download "$url" "$BIN/videoannotator-start.tmp"; then
    rm -f "$BIN/videoannotator-start.tmp"
    echo "Couldn't download VideoAnnotator. Check your internet connection and run this again." >&2
    exit 1
fi
chmod +x "$BIN/videoannotator-start.tmp"
mv -f "$BIN/videoannotator-start.tmp" "$BIN/videoannotator-start"
say "Installed videoannotator-start in $BIN."

case ":$PATH:" in
    *":$BIN:"*) ;;
    *) say "$BIN isn't on your PATH yet: open a new terminal, or run it as $BIN/videoannotator-start." ;;
esac

if [ "$shortcut" = 1 ]; then
    case "$(uname -s)" in
        Darwin)
            for dir in "$HOME/Desktop" "$HOME/Applications"; do
                [ -d "$dir" ] || continue
                printf '#!/bin/sh\nexec "%s/videoannotator-start" "$@"\n' "$BIN" > "$dir/Start VideoAnnotator.command"
                chmod +x "$dir/Start VideoAnnotator.command"
                say "Added \"Start VideoAnnotator\" to $dir."
                break
            done ;;
        *)
            entry="[Desktop Entry]
Type=Application
Name=Start VideoAnnotator
Comment=Start VideoAnnotator and open it in your browser
Exec=\"$BIN/videoannotator-start\"
Terminal=true
Categories=Science;Video;"
            apps="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
            mkdir -p "$apps"
            printf '%s\n' "$entry" > "$apps/videoannotator-start.desktop"
            desktop=$(xdg-user-dir DESKTOP 2>/dev/null || echo "$HOME/Desktop")
            if [ -d "$desktop" ]; then
                printf '%s\n' "$entry" > "$desktop/videoannotator-start.desktop"
                chmod +x "$desktop/videoannotator-start.desktop"
            fi
            say "Added \"Start VideoAnnotator\" to your applications." ;;
    esac
fi

say ""
say "To start VideoAnnotator, run: videoannotator-start"
