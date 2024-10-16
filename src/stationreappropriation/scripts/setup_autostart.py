#!/usr/bin/env python3
# stationreappropriation/scripts/setup_autostart.py

import os
import shutil
import subprocess
from pathlib import Path

def copy_start_script():
    script_path = Path(__file__).parent / "start_marimo.sh"
    dest_path = Path.home() / "start_marimo.sh"
    shutil.copy(script_path, dest_path)
    os.chmod(dest_path, 0o755)
    return dest_path

def create_systemd_service(script_path):
    service_content = f"""
        [Unit]
        Description=Démarrage automatique de Marimo
        After=network.target

        [Service]
        ExecStart={script_path}
        User={os.getlogin()}
        Environment=DISPLAY=:0

        [Install]
        WantedBy=multi-user.target
        """
    service_path = "/etc/systemd/system/marimo-autostart.service"
    with open(service_path, "w") as f:
        f.write(service_content)
    return service_path

def setup_systemd_service(service_name):
    subprocess.run(["sudo", "systemctl", "daemon-reload"])
    subprocess.run(["sudo", "systemctl", "enable", service_name])
    subprocess.run(["sudo", "systemctl", "start", service_name])

def main():
    try:
        print("Configuration de l'autostart de Marimo...")
        
        # Copie du script de démarrage
        start_script_path = copy_start_script()
        print(f"Script de démarrage copié vers : {start_script_path}")

        # Création du service systemd
        service_path = create_systemd_service(start_script_path)
        print(f"Service systemd créé : {service_path}")

        # Configuration et démarrage du service
        service_name = "marimo-autostart.service"
        setup_systemd_service(service_name)
        print(f"Service {service_name} activé et démarré")

        print("Configuration de l'autostart de Marimo terminée avec succès.")
    except Exception as e:
        print(f"Une erreur est survenue lors de la configuration : {e}")
        raise

if __name__ == "__main__":
    main()