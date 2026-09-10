#!/usr/bin/env bash
# Install openrouter-proxy as a systemd user service.
# Usage: OPENROUTER_API_KEY=sk-or-... ./install.sh [repo-url]
#        or: curl -fsSL <url>/install.sh | bash   (prompts for the key)

set -euo pipefail

REPO_URL="${1:-https://github.com/HubertasVin/openrouter-proxy.git}"
INSTALL_DIR="$HOME/.local/opt/openrouter-proxy"
CONFIG_DIR="$HOME/.config/openrouter-proxy"
SERVICE_NAME="openrouter-proxy"
PORT="8787"

if [ -z "${OPENROUTER_API_KEY:-}" ]; then
    if [ -t 0 ]; then
        read -r -p "OpenRouter API key: " OPENROUTER_API_KEY
    else
        echo "Error: OPENROUTER_API_KEY not set and stdin is not a terminal." >&2
        echo "Run: OPENROUTER_API_KEY=sk-or-... bash -c \"\$(curl -fsSL <url>/install.sh)\"" >&2
        exit 1
    fi
fi

if [ -z "$OPENROUTER_API_KEY" ]; then
    echo "Error: empty API key." >&2
    exit 1
fi

echo "==> Cloning/updating $REPO_URL -> $INSTALL_DIR"
if [ -d "$INSTALL_DIR/.git" ]; then
    git -C "$INSTALL_DIR" pull --ff-only
else
    mkdir -p "$(dirname "$INSTALL_DIR")"
    git clone "$REPO_URL" "$INSTALL_DIR"
fi

echo "==> Creating venv and installing dependencies"
python3 -m venv "$INSTALL_DIR/.venv"
"$INSTALL_DIR/.venv/bin/pip" install --quiet --upgrade pip
"$INSTALL_DIR/.venv/bin/pip" install --quiet -r "$INSTALL_DIR/requirements.txt"

echo "==> Writing config to $CONFIG_DIR/env"
mkdir -p "$CONFIG_DIR"
umask 077
cat > "$CONFIG_DIR/env" <<EOF
OPENROUTER_API_KEY=$OPENROUTER_API_KEY
PRIVACY_MODE=${PRIVACY_MODE:-prioritise_privacy}
PROXY_FOOTER=${PROXY_FOOTER:-1}
EOF
umask 022

echo "==> Installing systemd user service"
mkdir -p "$HOME/.config/systemd/user"
cat > "$HOME/.config/systemd/user/$SERVICE_NAME.service" <<EOF
[Unit]
Description=OpenRouter provider-filtering proxy
After=network-online.target

[Service]
Type=simple
WorkingDirectory=$INSTALL_DIR
EnvironmentFile=$CONFIG_DIR/env
ExecStart=$INSTALL_DIR/.venv/bin/uvicorn main:app --host 127.0.0.1 --port $PORT
Restart=on-failure

[Install]
WantedBy=default.target
EOF

systemctl --user daemon-reload
systemctl --user enable --now "$SERVICE_NAME"

echo "==> Done. Proxy listening on http://127.0.0.1:$PORT/v1"
echo "    Logs:     journalctl --user -u $SERVICE_NAME -f"
echo "    Stop:     systemctl --user stop $SERVICE_NAME"
echo "    Config:   $CONFIG_DIR/env (edit, then: systemctl --user restart $SERVICE_NAME)"