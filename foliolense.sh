#!/usr/bin/env bash
set -u

APP="$HOME/.local/share/foliolense"
PY="$APP/.venv/bin/python"
ENGINE="$APP/foliolense_crawl.py"
DB="$APP/index.sqlite"
LAST="$APP/last-folder.txt"

if [[ ! -x "$PY" || ! -f "$ENGINE" ]]; then
  echo "FolioLense is not installed. Open the downloaded FolioLense folder and run: ./install.sh"
  exit 1
fi

if [[ -t 1 && -n "${TERM:-}" ]]; then
  clear
fi
echo "FolioLense"
echo "=========="
echo "Search your PDFs and images on this computer."
echo "First time? Choose 1 to prepare documents for searching."
echo

while true; do
  echo "1) Add or update searchable documents"
  echo "2) Search documents"
  echo "3) Show document summary"
  echo "4) Quit"
  echo
  read -r -p "Choose [1-4]: " CHOICE || exit 0
  echo

  case "$CHOICE" in
    1)
      DEFAULT="$HOME/Documents"
      if [[ -f "$LAST" ]]; then
        DEFAULT="$(cat "$LAST")"
      fi

      read -r -p "Folder or file to add [$DEFAULT]: " FOLDER || exit 0
      FOLDER="${FOLDER:-$DEFAULT}"
      FOLDER="${FOLDER/#\~/$HOME}"

      if [[ ! -e "$FOLDER" ]]; then
        echo "File or folder not found: $FOLDER"
        echo
        continue
      fi

      printf '%s\n' "$FOLDER" > "$LAST"
      echo
      echo "Preparing documents for search: $FOLDER"
      echo "PDFs and images are supported. The first run downloads the search model."
      echo "You can stop safely with Ctrl+C and continue later."
      echo
      "$PY" "$ENGINE" crawl "$FOLDER" --db "$DB"
      echo
      ;;

    2)
      if [[ ! -f "$DB" ]]; then
        echo "No searchable documents yet. Choose 1 to add a folder or file."
        echo
        continue
      fi

      echo "Describe what you want to find, for example: shift schedules"
      read -r -p "Search for: " QUERY || exit 0
      if [[ -z "${QUERY//[[:space:]]/}" ]]; then
        echo "Enter some search words, or choose 4 to quit."
        echo
        continue
      fi

      echo
      "$PY" "$ENGINE" query "$QUERY" --db "$DB" --top 15 --interactive
      echo
      ;;

    3)
      if [[ ! -f "$DB" ]]; then
        echo "No document summary yet. Choose 1 to add a folder or file."
      else
        "$PY" "$ENGINE" stats --db "$DB"
      fi
      echo
      ;;

    4|q|Q)
      exit 0
      ;;

    *)
      echo "Choose 1, 2, 3, or 4."
      echo
      ;;
  esac
done
