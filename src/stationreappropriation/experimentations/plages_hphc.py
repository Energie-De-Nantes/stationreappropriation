import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np

    from pathlib import Path

    from electriflux.simple_reader import process_flux, iterative_process_flux
    return Path, mo


@app.cell
def _(Path, mo):
    folder_picker = mo.ui.file_browser(
        initial_path=Path('~/data/').expanduser(),
        selection_mode="directory",
        label="Sélectionnez le dossier de flux ENEDIS"
    )
    return (folder_picker,)


@app.cell(hide_code=True)
def _(folder_picker, mo):
    mo.md(
        f"""
    ## Sélection du dossier de flux ENEDIS
    {folder_picker}
    """
    )
    return


@app.cell(hide_code=True)
def _(folder_picker, mo):
    mo.stop(not folder_picker.value, "⚠️ Sélectionnez d'abord un dossier")

    selected_path = folder_picker.path(0)

    # Chemins des fichiers de clés
    key_file = selected_path / "clef_chiffrement.txt"
    iv_file = selected_path / "mot_de_passe.txt"

    # Chargement de la clé de chiffrement
    mo.stop(not key_file.exists(), f"❌ Fichier clé_chiffrement.txt introuvable dans {selected_path}")

    key_hex = key_file.read_text().strip()
    key = bytes.fromhex(key_hex)

    # Chargement du mot de passe (15 caractères)
    mo.stop(not iv_file.exists(), f"❌ Fichier mot_de_passe.txt introuvable dans {selected_path}")
    iv_raw = iv_file.read_text().strip()
    
    # TODO: Clarifier avec ENEDIS le format exact de l'IV
    # En attendant, padding simple à 16 bytes
    print(f"IV original: '{iv_raw}' (longueur: {len(iv_raw)})")
    iv = (iv_raw + '\0').encode('utf-8')[:16]
    print(f"IV utilisé: {iv.hex()}")
    return iv, key, selected_path, iv_raw


@app.cell(hide_code=True)
def _(iv, key, mo, selected_path, iv_raw):
    import zipfile
    from electriflux.utils import decrypt_file

    # Recherche des fichiers .zip
    zip_files = list(selected_path.glob("*.zip"))
    mo.stop(not zip_files, f"❌ Aucun fichier .zip trouvé dans {selected_path}")

    # Dossier de sortie
    output_path = selected_path / "extracted"
    output_path.mkdir(exist_ok=True)

    processed_files = []

    for zip_file in zip_files:
        try:
            print(f"\n📁 Traitement de {zip_file.name}")
            
            # Tentative avec electriflux (pour documentation)
            try:
                decrypted_path = decrypt_file(zip_file, key, iv)
                with zipfile.ZipFile(decrypted_path, 'r') as zip_ref:
                    zip_ref.extractall(output_path)
                print("✅ Décryptage réussi avec electriflux")
                processed_files.append(zip_file.name)
            except Exception as e:
                print(f"❌ Échec avec electriflux: {e}")
                print("💡 Vérifier les clés de chiffrement avec ENEDIS")

        except Exception as e:
            print(f"❌ Erreur lors du traitement de {zip_file.name}: {e}")
    
    print(f"\n📊 Résumé: {len(processed_files)} fichiers traités sur {len(zip_files)}")
    return


if __name__ == "__main__":
    app.run()
