import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""# SETUP""")
    return


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    from pandas.tseries.offsets import MonthBegin
    import numpy as np

    from pathlib import Path
    from babel.dates import format_date

    from electriflux.simple_reader import iterative_process_flux

    from stationreappropriation import load_prefixed_dotenv

    env = load_prefixed_dotenv(prefix='SR_')
    flux_path = Path('~/data/flux_enedis_v2/').expanduser()
    flux_path.mkdir(parents=True, exist_ok=True)
    return flux_path, iterative_process_flux, mo


@app.cell
def _(flux_path, iterative_process_flux):
    from electricore.inputs.flux import lire_flux_c15

    historique = lire_flux_c15(iterative_process_flux('C15', flux_path / 'C15'))
    historique = historique[sorted(historique.columns)]
    return (historique,)


@app.cell
def _(flux_path, iterative_process_flux):
    from electricore.inputs.flux import lire_flux_r151

    relevés = lire_flux_r151(iterative_process_flux('R151', flux_path / 'R151'))
    relevés
    return (relevés,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Ajout des points de rupture""")
    return


@app.cell
def _(historique):
    from electricore.core.périmètre import detecter_points_de_rupture as dpr

    ruptures = dpr(historique)
    ruptures
    return (ruptures,)


@app.cell
def _(ruptures):
    ruptures[ruptures["resume_modification"].fillna("") != ""]
    return


@app.cell
def _(mo):
    mo.md(r"""# Insertion événements de facturation""")
    return


@app.cell
def _(ruptures):
    from electricore.core.périmètre import inserer_evenements_facturation as ief
    etendu = ief(ruptures)
    return (etendu,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Calcul part Abonnement""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    On a deux choses à calculer ici. La quantité de produit abonnement journalier, par puissance. 
    Le Turpe fixe, qui dépend de la FTA et de la puissance.

    Pour cela, on va s'appuyer sur les événements survenus dans le périmètre, particulièrement ceux qui ont un impact sur le Turpe fixe, qui ont été taggués avec _impact_turpe_fixe = True_.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Construction des périodes d'abonnement""")
    return


@app.cell
def _(etendu):
    from electricore.core.abonnements import generer_periodes_abonnement as gpa
    periodes = gpa(etendu)
    # periodes.rename(columns={'Formule_Tarifaire_Acheminement': 'FTA'})
    periodes
    return (periodes,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Calcul du Turpe Fixe""")
    return


@app.cell
def _(periodes):
    from electricore.core.taxes.turpe import load_turpe_rules, ajouter_turpe_fixe
    règles = load_turpe_rules()
    periodes_turpe = ajouter_turpe_fixe(periodes, règles)
    periodes_turpe
    return (periodes_turpe,)


@app.cell
def _(periodes_turpe):
    periodes_turpe[periodes_turpe['mois_annee']=='avril 2025']
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# table d'index pour calcul des énergies""")
    return


@app.cell
def _(relevés, ruptures):
    evenements_impactants = ruptures[
        ruptures["impact_energie"] | ruptures["impact_turpe_variable"]
    ].copy()
    r = relevés.sort_values(["pdl", "Date_Releve"]).copy()
    return (evenements_impactants,)


@app.cell
def _(evenements_impactants):
    from electricore.core.périmètre import extraire_releves_evenements
    extraire_releves_evenements(evenements_impactants)
    return


if __name__ == "__main__":
    app.run()
