#!/bin/bash
# stationreappropriation/scripts/start_marimo.sh
# Fonction pour afficher et exécuter une commande
run_command() {
    echo "Exécution de la commande : $1"
    eval "$1"
}
DEST_DIR="$HOME/station_reappropriation"

# Create destination directory if it doesn't exist
mkdir -p "$DEST_DIR"

# Ajoutez ceci au début du script
LOG_FILE="$DEST_DIR/marimo_startup.log"
echo "$(date): Marimo startup script exécuté" >> "$LOG_FILE"

# Définir le chemin de l'environnement virtuel
VENV_PATH="$DEST_DIR/.venv/stationreappropriation_env"

# Créer l'environnement virtuel s'il n'existe pas
if [ ! -d "$VENV_PATH" ]; then
    echo "Création de l'environnement virtuel..."
    run_command "python3 -m venv \"$VENV_PATH\""
fi

# Activer l'environnement virtuel
source "$VENV_PATH/bin/activate"

# Mettre à jour pip
run_command "pip install --upgrade pip"

# Installer ou mettre à jour votre package et ses dépendances
run_command "pip install --upgrade stationreappropriation"

# Obtenir le chemin du package installé
PACKAGE_PATH=$(python -c "import stationreappropriation; import os; print(os.path.dirname(stationreappropriation.__file__))")

# Définir le chemin de votre application Marimo
MARIMO_APP_DIR="$PACKAGE_PATH/interfaces"

if [ -d "$MARIMO_APP_DIR" ]; then
    echo "Copie des fichiers .py depuis $MARIMO_APP_DIR vers $DEST_DIR"
    find "$MARIMO_APP_DIR" -name '*.py' -exec cp {} "$DEST_DIR" \;
    echo "Copie terminée."
    # Afficher le contenu du répertoire de destination
    echo "Contenu de $DEST_DIR:"
    ls -la "$DEST_DIR"
    # Ajouter le répertoire de destination au PYTHONPATH
    export PYTHONPATH="$DEST_DIR:$PYTHONPATH"
    echo "PYTHONPATH mis à jour: $PYTHONPATH"
else
    echo "Erreur : Le répertoire Marimo n'a pas été trouvé à $MARIMO_APP_DIR"
    deactivate
    exit 1
fi

# Lancer l'interface Marimo
run_command "marimo edit \"$DEST_DIR\""

# Désactiver l'environnement virtuel
deactivate