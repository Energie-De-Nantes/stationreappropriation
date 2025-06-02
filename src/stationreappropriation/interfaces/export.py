import marimo

__generated_with = "0.11.31"
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
    return get_pdls, pdls


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
def chargement_perimetre(flux_path, iterative_process_flux, pdls):
    from electricore.inputs.flux import lire_flux_c15

    historique = lire_flux_c15(iterative_process_flux('C15', flux_path / 'C15'))
    historique['Marque'] = historique['pdl'].isin(pdls['pdl']).apply(lambda x: 'EDN' if x else 'ZEL')
    historique = historique[sorted(historique.columns)]
    return historique, lire_flux_c15


@app.cell(hide_code=True)
def chargement_releves(flux_path, iterative_process_flux):
    from electricore.inputs.flux import lire_flux_r151

    relevés = lire_flux_r151(iterative_process_flux('R151', flux_path / 'R151'))
    return lire_flux_r151, relevés


@app.cell(hide_code=True)
def _(flux_path, iterative_process_flux, pdls):
    from electricore.inputs.flux import lire_flux_f1x

    factures_réseau = lire_flux_f1x(iterative_process_flux('F15', flux_path / 'F15'))
    factures_réseau['Marque'] = factures_réseau['pdl'].isin(pdls['pdl']).apply(lambda x: 'EDN' if x else 'ZEL')
    prestations = factures_réseau[factures_réseau['Nature_EV'] == 'Prestations et frais']
    return factures_réseau, lire_flux_f1x, prestations


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
def _(prestations):
    print(prestations.dtypes.to_frame(name="dtype").reset_index().rename(columns={"index": "column"}).to_markdown(index=False))
    return


@app.cell
def _(export_path, pd):
    pd.read_parquet(export_path / 'historique.parquet')
    return


@app.cell
def _(export_path, relevés):
    releves_1er = relevés[relevés["Date_Releve"].dt.day == 1]
    releves_1er.to_parquet(export_path / 'releves_1er.parquet')
    releves_1er
    return (releves_1er,)


@app.cell
def _():
    # # Crée des booléens indiquant la présence d'un avant ou après
    # historique["has_avant"] = historique["Avant_Id_Calendrier_Distributeur"].notna()
    # historique["has_apres"] = historique["Après_Id_Calendrier_Distributeur"].notna()
    # historique["has_both"] = historique["has_avant"] & historique["has_apres"]

    # # Groupe par événement et résume
    # count_summary = historique.groupby("Evenement_Declencheur").agg(
    #     total=("Evenement_Declencheur", "count"),
    #     nb_avant=("has_avant", "sum"),
    #     nb_apres=("has_apres", "sum"),
    #     nb_both=("has_both", "sum")
    # ).reset_index()
    # count_summary
    return


@app.cell
def _(historique):
    historique[(historique["has_both"])]# & (historique["Evenement_Declencheur"] == 'CMAT')]
    return


@app.cell
def _(historique):
    print(historique.dtypes.to_frame(name="dtype").reset_index().rename(columns={"index": "column"}).to_markdown(index=False))
    return


@app.cell
def _(historique, pd):
    # On trie d'abord la DataFrame par pdl et date d'événement
    périodes = historique.sort_values(['pdl', 'Ref_Situation_Contractuelle', 'Date_Evenement'])

    # Colonnes déclenchant une rupture contractuelle
    cols_impactantes = [
        'Ref_Situation_Contractuelle',
        'Formule_Tarifaire_Acheminement',
        'Puissance_Souscrite',
    ]

    # Identifier les ruptures sur les colonnes clés
    périodes['rupture'] = périodes.groupby('pdl')[cols_impactantes].apply(
        lambda x: x.ne(x.shift()).any(axis=1)
    ).reset_index(level=0, drop=True)

    # Périodes principales (1er du mois au 1er du mois suivant)
    périodes['debut_periode_principale'] = périodes['Date_Evenement'].dt.to_period('M').dt.start_time
    périodes['fin_periode_principale'] = (périodes['debut_periode_principale'] + pd.offsets.MonthBegin(1))

    # Construction des périodes intermédiaires tenant compte des ruptures
    periodes_calculees = []
    for pdl, groupe in périodes.groupby('pdl'):
        groupe = groupe.reset_index(drop=True)
        periode_debut = groupe.loc[0, 'debut_periode_principale'].tz_localize('Europe/Paris')
    
        for i, row in groupe.iterrows():
            rupture = row['rupture']
            date_evenement = row['Date_Evenement']

            if rupture and date_evenement > periode_debut:
                periodes_calculees.append({
                    'pdl': pdl,
                    'periode_debut': periode_debut,
                    'periode_fin': date_evenement,
                    **{col: row[col] for col in cols_impactantes}
                })
                periode_debut = date_evenement

        # Dernière période jusqu'à la fin du mois en cours
        periodes_calculees.append({
            'pdl': pdl,
            'periode_debut': periode_debut,
            'periode_fin': groupe.loc[0, 'fin_periode_principale'],
            **{col: groupe.iloc[-1][col] for col in cols_impactantes}
        })

    # Finaliser le DataFrame résultat
    periodes_calculees_df = pd.DataFrame(periodes_calculees)
    from babel.dates import format_date

    # Appliquer format_date ligne par ligne
    periodes_calculees_df['mois_annee'] = periodes_calculees_df['periode_debut'].apply(
        lambda d: format_date(d, format="LLLL yyyy", locale="fr_FR")
    )
    periodes_calculees_df = periodes_calculees_df.sort_values(['pdl','Ref_Situation_Contractuelle', 'periode_debut']).reset_index(drop=True)
    periodes_calculees_df
    return (
        cols_impactantes,
        date_evenement,
        format_date,
        groupe,
        i,
        pdl,
        periode_debut,
        periodes_calculees,
        periodes_calculees_df,
        périodes,
        row,
        rupture,
    )


if __name__ == "__main__":
    app.run()
