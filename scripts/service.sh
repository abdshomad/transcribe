#!/usr/bin/env bash
# Transcribe Service Manager CLI
set -e

SERVICE_NAME="transcribe.service"

case "${1:-status}" in
  start)
    systemctl --user start "${SERVICE_NAME}"
    echo "Started ${SERVICE_NAME}"
    systemctl --user status "${SERVICE_NAME}" --no-pager
    ;;
  stop)
    systemctl --user stop "${SERVICE_NAME}"
    echo "Stopped ${SERVICE_NAME}"
    ;;
  restart)
    systemctl --user daemon-reload
    systemctl --user restart "${SERVICE_NAME}"
    echo "Restarted ${SERVICE_NAME}"
    sleep 1
    systemctl --user status "${SERVICE_NAME}" --no-pager
    ;;
  status)
    systemctl --user status "${SERVICE_NAME}" --no-pager
    ;;
  logs)
    journalctl --user -u "${SERVICE_NAME}" -f -n 50
    ;;
  enable)
    systemctl --user daemon-reload
    systemctl --user enable "${SERVICE_NAME}"
    echo "Enabled ${SERVICE_NAME} on user login"
    ;;
  disable)
    systemctl --user disable "${SERVICE_NAME}"
    echo "Disabled ${SERVICE_NAME}"
    ;;
  *)
    echo "Usage: $0 {start|stop|restart|status|logs|enable|disable}"
    exit 1
    ;;
esac
