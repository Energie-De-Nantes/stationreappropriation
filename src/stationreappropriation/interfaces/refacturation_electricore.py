import marimo

__generated_with = "0.11.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np

    from pathlib import Path

    from electriflux.simple_reader import process_flux

    from stationreappropriation import load_prefixed_dotenv

    env = load_prefixed_dotenv(prefix='SR_')
    flux_path = Path('~/data/flux_enedis_v2/').expanduser()
    flux_path.mkdir(parents=True, exist_ok=True)
    return Path, env, flux_path, load_prefixed_dotenv, mo, np, pd, process_flux


@app.cell
def _(env, flux_path, mo):
    from stationreappropriation.marimo_utils import download_with_marimo_progress as _dl

    _processed, _errors = _dl(env, ['F15', 'F12'], flux_path)

    mo.md(f"Processed #{len(_processed)} files, with #{len(_errors)} erreurs")
    return


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
    return f12, f15


@app.cell
def _(f12, f15, pd, pdls):
    from electricore.inputs.flux import lire_flux_f1x
    factures_réseau = lire_flux_f1x(pd.concat([f12, f15], ignore_index=True))
    factures_réseau['Marque'] = factures_réseau['pdl'].isin(pdls['pdl']).apply(lambda x: 'EDN' if x else 'ZEL')
    factures_réseau
    return factures_réseau, lire_flux_f1x


@app.cell
def _():
    def regroupement_mensuel_par_marque(df):
        factures = df.copy()
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


if __name__ == "__main__":
    app.run()
