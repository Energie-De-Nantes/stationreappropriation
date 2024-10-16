import subprocess

def remove_service(service_name):
    try:
        # Arrêter le service
        subprocess.run(["sudo", "systemctl", "stop", service_name], check=True)
        
        # Désactiver le service
        subprocess.run(["sudo", "systemctl", "disable", service_name], check=True)
        
        # Supprimer le fichier de service
        service_path = f"/etc/systemd/system/{service_name}"
        subprocess.run(["sudo", "rm", service_path], check=True)
        
        # Recharger systemd
        subprocess.run(["sudo", "systemctl", "daemon-reload"], check=True)
        
        print(f"Le service {service_name} a été supprimé avec succès.")
    except subprocess.CalledProcessError as e:
        print(f"Erreur lors de la suppression du service : {e}")
        raise

def main():
    service_name = "marimo-autostart.service"
    remove_service(service_name)

if __name__ == "__main__":
    main()