#!/usr/bin/env python3
# stationreappropriation/scripts/setup_autostart.py

import os
import pwd
import shutil
import subprocess
from pathlib import Path

def copy_start_script():
    script_path = Path(__file__).parent / "start_marimo.sh"
    dest_path = Path.home() / "station_reappropriation" / "start_marimo.sh"
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(script_path, dest_path)
    os.chmod(dest_path, 0o755)
    return dest_path

def create_systemd_service(script_path):
    username = pwd.getpwuid(os.getuid()).pw_name
    
    service_content = f"""
    [Unit]
    Description=Démarrage automatique de Marimo
    After=graphical.target
    Wants=graphical.target

    [Service]
    Type=simple
    ExecStart={script_path}
    User={username}
    Environment=DISPLAY=:0
    Environment=XAUTHORITY=/home/{username}/.Xauthority
    Restart=on-failure
    RestartSec=5s

    [Install]
    WantedBy=graphical.target
    """
    return service_content


def update_or_create_service(service_name, service_content):
    service_path = f"/etc/systemd/system/{service_name}"
    
    # Arrêter le service s'il existe
    subprocess.run(["sudo", "systemctl", "stop", service_name], check=False)
    
    # Écrire ou remplacer le fichier de service
    subprocess.run(["sudo", "tee", service_path], input=service_content.encode(), check=True)
    
    # Recharger systemd, activer et démarrer le service
    subprocess.run(["sudo", "systemctl", "daemon-reload"], check=True)
    subprocess.run(["sudo", "systemctl", "enable", service_name], check=True)
    subprocess.run(["sudo", "systemctl", "start", service_name], check=True)

def main():
    try:
        print("Configuration de l'autostart de Marimo...")
        
        start_script_path = copy_start_script()
        print(f"Script de démarrage copié vers : {start_script_path}")

        service_content = create_systemd_service(start_script_path)
        service_name = "marimo-autostart.service"
        
        update_or_create_service(service_name, service_content)
        
        print(f"Service {service_name} mis à jour/créé, activé et démarré")
        
    except subprocess.CalledProcessError as e:
        print(f"Erreur lors de l'exécution d'une commande : {e}")
        raise
    except Exception as e:
        print(f"Une erreur est survenue lors de la configuration : {e}")
        raise

if __name__ == "__main__":
    main()