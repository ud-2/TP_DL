#!/bin/bash

# Nom du fichier de sortie
OUTPUT_FILE="t.txt"

# Effacer le fichier s'il existe déjà
> $OUTPUT_FILE

echo "Génération du bundle dans $OUTPUT_FILE..."

# Liste des fichiers à exclure (regex)
# On exclut le venv, git, fichiers binaires et lockfiles
EXCLUDE_PATTERN="venv|\.git|\.pyc|__pycache__|\.env|\.DS_Store|\.pyo|\.pyd|\.so|LICENSE|t.txt|t.sh"

# Parcourir les fichiers
# On prend les fichiers texte courants en Python (py, md, txt, json, yaml, etc.)
find . -type f -not -path '*/.*' | grep -vE "$EXCLUDE_PATTERN" | while read -r file; do
    echo "Ajout de : $file"
    echo "================================================" >> $OUTPUT_FILE
    echo "FILE: $file" >> $OUTPUT_FILE
    echo "================================================" >> $OUTPUT_FILE
    cat "$file" >> $OUTPUT_FILE
    echo -e "\n\n" >> $OUTPUT_FILE
done

echo "Terminé ! Vous pouvez envoyer $OUTPUT_FILE à l'IA."
