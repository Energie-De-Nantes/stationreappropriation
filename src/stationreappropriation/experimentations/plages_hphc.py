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
    return Path, mo, pd


@app.cell
def _(Path, mo):
    csv_folder_picker = mo.ui.file_browser(
        initial_path=Path('~/data/').expanduser(),
        selection_mode="directory", 
        label="Sélectionnez le dossier contenant les CSV à concaténer"
    )
    return (csv_folder_picker,)


@app.cell(hide_code=True)
def _(csv_folder_picker, mo):
    mo.md(
        f"""
    ## Sélection du dossier CSV
    {csv_folder_picker}
    """
    )
    return


@app.cell(hide_code=True)
def _(csv_folder_picker, mo, pd):
    mo.stop(not csv_folder_picker.value, "⚠️ Sélectionnez le dossier CSV")

    csv_path = csv_folder_picker.path(0)
    csv_files = list(csv_path.glob("*.csv"))

    mo.stop(not csv_files, f"❌ Aucun fichier CSV trouvé dans {csv_path}")

    print(f"📁 Fichiers CSV trouvés : {len(csv_files)}")
    for csv_file in csv_files:
        print(f"  - {csv_file.name}")

    # Lecture et concaténation de tous les CSV
    dataframes = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file, sep=';')
            print(f"✅ {csv_file.name}: {len(df)} lignes")
            dataframes.append(df)
        except Exception as e:
            print(f"❌ Erreur lecture {csv_file.name}: {e}")

    if dataframes:
        concatenated_df = pd.concat(dataframes, ignore_index=True)
        print(f"\n📊 Données concaténées: {len(concatenated_df)} lignes, {len(concatenated_df.columns)} colonnes")
    else:
        concatenated_df = pd.DataFrame()
        print("❌ Aucune donnée à concaténer")

    return (concatenated_df,)


@app.cell(hide_code=True)
def _(concatenated_df, mo):
    mo.stop(concatenated_df.empty, "❌ Aucune donnée disponible")

    mo.md(f"""
    ## Aperçu des données concaténées
    **{len(concatenated_df)} lignes** - **{len(concatenated_df.columns)} colonnes**
    """)
    return


@app.cell
def _(concatenated_df):
    concatenated_df.sort_values(by=['OFFRE_FRN_HC', 'DATE_BASCULE'])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
