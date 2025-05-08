#!/usr/bin/env bash
#
# Launch script for TinyLCM examples (robust)

set -euo pipefail

# Absolutes Basis-Verzeichnis ermitteln
BASE="$(cd "$(dirname "$0")" && pwd)"
cd "$BASE"

# PYTHONPATH setzen
export PYTHONPATH="$BASE:$BASE/examples:$BASE/examples/utils:$BASE/tinylcm:$PYTHONPATH"

# Szenarien sammeln
declare -a SCRIPTS
declare -a NAMES

i=1
for scenario_dir in "$BASE/examples"/scenario*; do
  [ -d "$scenario_dir" ] || continue
  # Nimm das erste main_*.py
  main=$(find "$scenario_dir" -maxdepth 1 -type f -name 'main_*.py' | head -n1)
  if [ -n "$main" ]; then
    SCRIPTS+=("$main")
    NAMES+=("$(basename "$scenario_dir")")
    printf "  [%d] %s\n" "$i" "${NAMES[-1]}"
    ((i++))
  fi
done

total=${#SCRIPTS[@]}
if (( total == 0 )); then
  echo "Error: No scenarios found in $BASE/examples"
  exit 1
fi

echo
# Auswahl-Prompt
read -p "Select scenario to run (1-${total}): " sel

# Validierung via integer-Vergleich
if ! [[ "$sel" =~ ^[0-9]+$ ]]; then
  echo "Invalid input: not a number"
  exit 1
fi
if (( sel < 1 || sel > total )); then
  echo "Invalid selection: choose between 1 and ${total}"
  exit 1
fi

chosen="${SCRIPTS[$((sel-1))]}"
echo
echo "▶ Running scenario [${sel}] ${NAMES[$((sel-1))]}"
python3 "$chosen"
