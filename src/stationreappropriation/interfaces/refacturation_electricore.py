import marimo

__generated_with = "0.11.31"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np

    from pathlib import Path

    from electriflux.simple_reader import process_flux, iterative_process_flux

    from stationreappropriation import load_prefixed_dotenv

    env = load_prefixed_dotenv(prefix='SR_')
    flux_path = Path('~/data/flux_enedis_v2/').expanduser()
    flux_path.mkdir(parents=True, exist_ok=True)
    return (
        Path,
        env,
        flux_path,
        iterative_process_flux,
        load_prefixed_dotenv,
        mo,
        np,
        pd,
        process_flux,
    )


@app.cell
def _(env, flux_path, iterative_process_flux, mo):
    from stationreappropriation.marimo_utils import download_with_marimo_progress as _dl

    _processed, _errors = _dl(env, ['F15', 'F12'], flux_path)

    i_f15 = iterative_process_flux('F15', flux_path / 'F15')
    i_f12 = iterative_process_flux('F12', flux_path / 'F12')
    mo.md(f"Processed #{len(_processed)} files, with #{len(_errors)} erreurs")
    i_f15
    return i_f12, i_f15


@app.cell
def _(env, pd):
    from stationreappropriation.odoo import get_pdls
    pdls = get_pdls(env)
    _local = pd.DataFrame({
        'sale.order_id': [0, 0, 0],  # Exemple d'identifiant de commande
        'pdl': ['14295224261882', '50070117855585', '50000508594660']           # Exemple de PDL
    })

    # Ajouter la nouvelle ligne à la dataframe avec pd.concat
    pdls = pd.concat([pdls, _local], ignore_index=True)
    return get_pdls, pdls


@app.cell
def _(flux_path, process_flux):
    f15 = process_flux('F15', flux_path / 'F15')
    f12 = process_flux('F12', flux_path / 'F12')
    f15
    return f12, f15


@app.cell
def _(f12, f15, pd, pdls):
    from electricore.inputs.flux import lire_flux_f1x
    factures_réseau = lire_flux_f1x(pd.concat([f12, f15], ignore_index=True))
    factures_réseau['Marque'] = factures_réseau['pdl'].isin(pdls['pdl']).apply(lambda x: 'EDN' if x else 'ZEL')
    factures_réseau["Mois"] = factures_réseau["Date_Facture"].dt.to_period("M")
    duplicates = factures_réseau[factures_réseau.duplicated()]
    factures_réseau = factures_réseau.drop_duplicates()
    return duplicates, factures_réseau, lire_flux_f1x


@app.cell
def _():
    # mo.accordion({'Factures réseau': factures_réseau, 'Doublons': duplicates})
    return


@app.cell
def _():
    ## Filtrage mensuel
    return


@app.cell
def _():
    # _options = {str(m):m for m in factures_réseau['Mois'].unique().tolist()}
    # multiselect = mo.ui.multiselect(options=_options)
    # multiselect
    return


@app.cell
def _():
    # _selected = factures_réseau[factures_réseau["Mois"].isin(multiselect.value)]
    # duplicates = _selected[_selected.duplicated(keep=False)]
    # duplicates
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Regroupement Mensuel par marque""")
    return


@app.cell
def _():
    def regroupement_mensuel_par_marque(df):
        factures = df.copy()
        if "Mois" not in df.columns:
            factures["Mois"] = factures["Date_Facture"].dt.to_period("M")

        df_grouped = (
            factures
            .groupby(["Mois", "Marque", "Taux_TVA_Applicable", "Source"])["Montant_HT"]
            .sum()
            .unstack(["Marque", "Taux_TVA_Applicable","Source"])  # Création de colonnes pour chaque combinaison
            .fillna(0)  # Remplacer les NaN par 0
            .reset_index()
        )
        df_grouped.columns = ['_'.join(map(str, col)) if isinstance(col, tuple) else col for col in df_grouped.columns]
        return df_grouped
    return (regroupement_mensuel_par_marque,)


@app.cell
def _(factures_réseau, regroupement_mensuel_par_marque):
    regroupement_mensuel_par_marque(factures_réseau)
    return


@app.cell
def _(factures_réseau, regroupement_mensuel_par_marque):
    résumé =regroupement_mensuel_par_marque(factures_réseau[factures_réseau['Marque']=='ZEL'])

    résumé['Total HT'] = résumé[[c for c in résumé.columns if c.startswith('ZEL')]].sum(axis=1)
    résumé['Total TVA'] = résumé[[c for c in résumé.columns if '20.0' in c]].sum(axis=1) * 0.2
    résumé['Total TTC'] = résumé['Total HT'] + résumé['Total TVA']

    print(résumé['ZEL_20.0_Flux_F12'].sum())
    résumé.round(2)
    return (résumé,)


@app.cell
def _(mo):
    mo.md(
        """
        ## Vérification des factures globales
        On a constaté des lignes dupliquées dans les fichiers détails, 
        on va regrouper les lignes par factures, sans les duplicatas puis afficher les factures ou il y a duplicatas
        """
    )
    return


@app.cell
def _():
    def regroupement_factures_globales(df):
        factures = df.copy()

        df_grouped = (
            factures
            .groupby(["Num_Facture", "Marque", "Taux_TVA_Applicable", "Source"])["Montant_HT"]
            .sum()
            .unstack(["Marque", "Taux_TVA_Applicable","Source"])  # Création de colonnes pour chaque combinaison
            .fillna(0)  # Remplacer les NaN par 0
            .reset_index()
        )
        df_grouped.columns = ['_'.join(map(str, col)) if isinstance(col, tuple) else col for col in df_grouped.columns]
        return df_grouped
    return (regroupement_factures_globales,)


@app.cell
def _(factures_réseau, regroupement_factures_globales):
    factures_globales = regroupement_factures_globales(factures_réseau)
    factures_globales
    return (factures_globales,)


@app.cell
def _(duplicates, factures_globales):
    factures_globales[factures_globales['Num_Facture__'].isin(duplicates['Num_Facture'])]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Turpe Facturé mensuel

        On vient décomposer chaque ligne sur les mois qu'elle couvre. Ces mois sont ceux inclus ou partiellement inclus dans la période définie par (Date_Debut, Date_Fin). Le montant HT de chaque ligne est amputé aux mois qu'elle couvre au pro-rata de jours du mois sur le total de jours.
        """
    )
    return


@app.cell
def _(factures_réseau, pd):
    df = factures_réseau.copy()
    # Création de toutes les dates de début et fin de mois pour chaque ligne
    df["Mois_Debut"] = df["Date_Debut"].apply(lambda x: x.replace(day=1))
    df["Mois_Fin"] = df["Date_Fin"].apply(lambda x: x.replace(day=1))

    # Création d'une liste de tous les mois impliqués
    all_months = pd.date_range(df["Mois_Debut"].min(), df["Mois_Fin"].max(), freq='MS')

    # Création d'un DataFrame multi-index avec chaque facture et chaque mois concerné
    factures_expanded = df.loc[df.index.repeat(len(all_months))]
    factures_expanded["Mois"] = list(all_months) * len(df)

    # Filtrer uniquement les mois dans l'intervalle de chaque facture
    factures_expanded = factures_expanded[
        (factures_expanded["Mois"] >= factures_expanded["Mois_Debut"]) &
        (factures_expanded["Mois"] <= factures_expanded["Mois_Fin"])
    ]

    # Calcul des dates effectives pour chaque mois
    factures_expanded["Debut_Mois_Reel"] = factures_expanded[["Date_Debut", "Mois"]].max(axis=1)
    factures_expanded["Fin_Mois_Reel"] = factures_expanded.apply(
        lambda row: min(row["Date_Fin"], row["Mois"] + pd.offsets.MonthEnd(0)), axis=1
    )
    # Calcul du nombre de jours dans chaque mois et prorata
    factures_expanded["Jours_Mois"] = (factures_expanded["Fin_Mois_Reel"] - factures_expanded["Debut_Mois_Reel"]).dt.days + 1
    factures_expanded["Jours_Totaux"] = (factures_expanded["Date_Fin"] - factures_expanded["Date_Debut"]).dt.days + 1
    factures_expanded["Montant_HT_Mensuel"] = factures_expanded["Montant_HT"] * (factures_expanded["Jours_Mois"] / factures_expanded["Jours_Totaux"])

    factures_expanded
    return all_months, df, factures_expanded


@app.cell
def _(factures_expanded):
    # Sélection des colonnes finales
    df_result = factures_expanded[["Num_Facture", "Taux_TVA_Applicable", "Marque", "Mois", "Montant_HT_Mensuel", "Source"]].rename(columns={"Montant_HT_Mensuel": "Montant_HT"})


    df_result
    return (df_result,)


@app.cell
def _(mo):
    mo.md(r"""## On regroupe par mois, marque, et TVA""")
    return


@app.cell(hide_code=True)
def _(df_result, factures_réseau, np):
    def regroupement_mensuel_complet(df):
        factures = df.copy()

        df_grouped = (
            factures
            .groupby(["Mois", "Marque", "Taux_TVA_Applicable"])["Montant_HT"]
            .sum()
            .unstack(["Marque", "Taux_TVA_Applicable"])  # Création de colonnes pour chaque combinaison
            .fillna(0)  # Remplacer les NaN par 0
            .reset_index()
        )
        df_grouped.columns = ['_'.join(map(str, col)) if isinstance(col, tuple) else col for col in df_grouped.columns]
        return df_grouped
    remensualisé = (
        regroupement_mensuel_complet(df_result)

        .set_index('Mois_')
    )
    # remensualisé['total'] = remensualisé.sum(axis=1)
    remensualisé = remensualisé.round(2)


    remensualisé['Total HT ZEL'] = remensualisé[[c for c in remensualisé.columns if c.startswith('ZEL')]].sum(axis=1)
    remensualisé['Total HT EDN'] = remensualisé[[c for c in remensualisé.columns if c.startswith('EDN')]].sum(axis=1)
    remensualisé['Total HT'] = remensualisé['Total HT ZEL'] + remensualisé['Total HT EDN']
    remensualisé['Total TVA'] = remensualisé[[c for c in remensualisé.columns if '20.0' in c]].sum(axis=1) * 0.2
    remensualisé['Total TTC'] = remensualisé['Total HT'] + remensualisé['Total TVA']

    assert np.isclose(
        remensualisé['Total HT'].sum(), 
        factures_réseau['Montant_HT'].sum(), 
        atol=0.02  # tolérance absolue (ex : 2 centime)
    )

    remensualisé.round(2)
    return regroupement_mensuel_complet, remensualisé


@app.cell
def _(mo):
    mo.md("""## Regroupement par facture""")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
