import marimo

__generated_with = "0.11.31"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import altair as alt
    import matplotlib.pyplot as plt
    from pathlib import Path

    where_to_look = Path("~/data/M023").expanduser()
    file_browser = mo.ui.file_browser(
        initial_path=where_to_look, multiple=True, filetypes=['.csv', '.json']
    )
    file_browser
    return Path, alt, file_browser, mo, np, pd, plt, where_to_look


@app.cell(hide_code=True)
def _(file_browser, pd):
    _timeserie_df_list = []
    for _file in file_browser.value:
        _timeserie_df_list.append(pd.read_csv(_file.path, sep=';', parse_dates=['Date de début', 'Date de fin', 'Début de la mesure', 'Fin de la mesure']))

    # Concaténer tous les dataframes de timeserie
    if _timeserie_df_list:
        timeserie_df = pd.concat(_timeserie_df_list, ignore_index=True)
    else:
        timeserie_df = pd.DataFrame()

    if not timeserie_df.empty:
        # Ajout de la Marque
        timeserie_df['pdl'] = timeserie_df['Identifiant PRM'].astype(str).str.zfill(14)

    timeserie_df['nb_jours'] = (timeserie_df['Fin de la mesure'].dt.normalize() - timeserie_df['Début de la mesure'].dt.normalize()).dt.days
    timeserie_df
    return (timeserie_df,)


@app.cell
def _(timeserie_df):
    timeserie_df.columns
    return


@app.cell
def _(timeserie_df):
    len(timeserie_df['Identifiant PRM'].unique())
    return


@app.cell(hide_code=True)
def _(np, timeserie_df):
    # Liste des cadrans que tu veux traiter
    cadrans = ['BASE', 'HP', 'HC']

    # Agrégation
    df_grouped = (
        timeserie_df[timeserie_df['Code Grille'] == 'F']
        .groupby(['Identifiant PRM', 'Identifiant classe temporelle'])[['Quantité', 'nb_jours']]
        .sum()
        .div({'Quantité': 1000, 'nb_jours': 1})
        .unstack()
        .reset_index()
    )

    df_grouped.columns = [f'{col[1]}_{col[0]}' for col in df_grouped.columns]

    # Trouver la première colonne *_nb_jours
    jours_cols = [col for col in df_grouped.columns if col.endswith('_nb_jours')]

    # Créer une seule colonne nb_jours à partir de la première non-nan
    df_grouped['nb_jours'] = df_grouped[jours_cols].bfill(axis=1).iloc[:, 0]

    # Supprimer les colonnes intermédiaires
    df_grouped = df_grouped.drop(columns=jours_cols)

    df_grouped = df_grouped.rename(columns={
        'BASE_Quantité': 'BASE',
        'HP_Quantité': 'HP',
        'HC_Quantité': 'HC',
        '_Identifiant PRM': 'Identifiant PRM'
    })

    # Garder les colonnes utiles
    df_grouped = df_grouped[['Identifiant PRM', 'BASE', 'HP', 'HC', 'nb_jours']]


    # Ajout des colonnes divisées par 12
    for cadran in cadrans:
        if cadran in df_grouped.columns:
            df_grouped[f'{cadran}_mensuel'] = np.ceil(
                df_grouped[cadran] / (df_grouped['nb_jours'] / 30.44)
            )
    df_grouped = df_grouped.sort_values(by='nb_jours', ascending=False).reset_index(drop=True)
    df_grouped
    return cadran, cadrans, df_grouped, jours_cols


if __name__ == "__main__":
    app.run()
