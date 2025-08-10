import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def choix_mois_facturation():
    import marimo as mo
    import pandas as pd
    import numpy as np

    from pathlib import Path

    from electriflux.simple_reader import process_flux, iterative_process_flux

    from stationreappropriation import load_prefixed_dotenv

    env = load_prefixed_dotenv(prefix='SR_')
    flux_path = Path('~/data/flux_enedis_v2/').expanduser()
    flux_path.mkdir(parents=True, exist_ok=True)
    return Path, env, flux_path, iterative_process_flux, mo, pd, process_flux


@app.cell(hide_code=True)
def _(env, pd):
    from stationreappropriation.odoo import get_pdls
    pdls = get_pdls(env)
    _local = pd.DataFrame({
        'sale.order_id': [0, 0, 0],  # Exemple d'identifiant de commande
        'pdl': ['14295224261882', '50070117855585', '50000508594660']           # Exemple de PDL
    })

    # Ajouter la nouvelle ligne à la dataframe avec pd.concat
    pdls = pd.concat([pdls, _local], ignore_index=True)
    return (pdls,)


@app.cell
def _(Path, pdls):
    pdls['pdl'].to_csv(Path("~/data/pdl.csv").expanduser(), index=False)
    return


@app.cell
def _(mo):
    mo.md(r"""# Chargement des flux""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Téléchargement et extraction des zip""")
    return


@app.cell(hide_code=True)
def telechargement_flux(env, flux_path, mo):
    from stationreappropriation.marimo_utils import download_with_marimo_progress as _dl
    _processed, _errors = _dl(env, ['R15', 'R151', 'C15', 'F15', 'F12'], flux_path)
    mo.md(f"Processed #{len(_processed)} files, with #{len(_errors)} erreurs")
    return


@app.cell(hide_code=True)
def _():
    ## Interprétation des flux
    return


@app.cell(hide_code=True)
def chargement_perimetre(flux_path, pdls, process_flux):
    from electricore.inputs.flux import lire_flux_c15

    historique = lire_flux_c15(process_flux('C15', flux_path / 'C15'))
    historique['Marque'] = historique['pdl'].isin(pdls['pdl']).apply(lambda x: 'EDN' if x else 'ZEL')
    historique = historique[sorted(historique.columns)]
    return (historique,)


@app.cell(hide_code=True)
def chargement_releves(flux_path, process_flux):
    from electricore.inputs.flux import lire_flux_r151

    relevés = lire_flux_r151(process_flux('R151', flux_path / 'R151'))
    return (relevés,)


@app.cell(hide_code=True)
def _(flux_path, iterative_process_flux, pdls):
    from electricore.inputs.flux import lire_flux_f1x

    factures_réseau = lire_flux_f1x(iterative_process_flux('F15', flux_path / 'F15'))
    factures_réseau['Marque'] = factures_réseau['pdl'].isin(pdls['pdl']).apply(lambda x: 'EDN' if x else 'ZEL')
    prestations = factures_réseau[factures_réseau['Nature_EV'] == 'Prestations et frais']
    return (prestations,)


@app.cell
def _(flux_path, process_flux):
    process_flux('R15', flux_path / 'R15')
    return


@app.cell(hide_code=True)
def visualisation_donnees_metier(historique, mo, prestations, relevés):
    mo.accordion({"relevés": relevés, 'historique': historique, 'prestations': prestations}, lazy=True)
    return


@app.cell
def _(mo):
    mo.md(r"""# Export""")
    return


@app.cell
def export(Path, historique, prestations, relevés):
    export_path = Path('~/data/export_flux/').expanduser()
    export_path.mkdir(parents=True, exist_ok=True)

    relevés.to_parquet(export_path / 'releves.parquet')
    historique.to_parquet(export_path / 'historique.parquet')
    prestations.to_parquet(export_path / 'prestations.parquet')
    return (export_path,)


@app.cell
def _(export_path, pd):
    pd.read_parquet(export_path / 'historique.parquet')
    return


@app.cell
def _(export_path, relevés):
    releves_1er = relevés[relevés["Date_Releve"].dt.day == 1]
    releves_1er.to_parquet(export_path / 'releves_1er.parquet')
    releves_1er
    return


@app.cell
def _(historique):
    len(historique['pdl'].unique())
    return


if __name__ == "__main__":
    app.run()
