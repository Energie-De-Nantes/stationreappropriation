import marimo

__generated_with = "0.14.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np

    from datetime import date
    from pathlib import Path

    from electriflux.simple_reader import process_flux, iterative_process_flux

    from stationreappropriation import load_prefixed_dotenv

    env = load_prefixed_dotenv(prefix='SR_')
    flux_path = Path('~/data/flux_enedis_v2/').expanduser()
    flux_path.mkdir(parents=True, exist_ok=True)
    return Path, env, flux_path, mo, pd, process_flux


@app.cell
def _(env, flux_path, mo):
    from stationreappropriation.marimo_utils import download_with_marimo_progress as _dl

    _processed, _errors = _dl(env, ['R15', 'R151', 'C15', 'F15', 'F12'], flux_path)

    mo.md(f"Processed #{len(_processed)} files, with #{len(_errors)} erreurs")
    return


@app.cell(hide_code=True)
def choix_mois_facturation(mo):
    options = {"T1": 1, "T2": 2, "T3": 3, "T4": 4}
    radio = mo.ui.radio(options=options, label='Choisi le Trimestre', value="T1")
    radio
    return (radio,)


@app.cell(hide_code=True)
def choix_dates_facturation(mo, radio):
    from stationreappropriation.utils import gen_trimester_dates
    _default_start, _default_end = gen_trimester_dates(radio.value)
    start_date_picker = mo.ui.date(value=_default_start)
    end_date_picker = mo.ui.date(value=_default_end)
    mo.md(
        f"""
        Date de début {start_date_picker} et de fin {end_date_picker}\n
        """
    )
    return end_date_picker, start_date_picker


@app.cell(hide_code=True)
def conversion_dates(end_date_picker, pd, start_date_picker):
    from zoneinfo import ZoneInfo
    PARIS_TZ = ZoneInfo("Europe/Paris")
    deb = pd.to_datetime(start_date_picker.value).tz_localize(PARIS_TZ)
    fin = pd.to_datetime(end_date_picker.value).tz_localize(PARIS_TZ)
    return PARIS_TZ, deb, fin


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
def chargement_perimetre(deb, fin, flux_path, pdls, process_flux):
    from electricore.inputs.flux import lire_flux_c15

    historique = lire_flux_c15(process_flux('C15', flux_path / 'C15'))
    historique['Marque'] = historique['pdl'].isin(pdls['pdl']).apply(lambda x: 'EDN' if x else 'ZEL')
    historique = historique[sorted(historique.columns)]
    historique = historique[historique['Marque'] == 'EDN']

    from electricore.core.périmètre.fonctions import extraire_historique_à_date, extraire_modifications_impactantes

    mci = extraire_modifications_impactantes(deb=deb, historique=extraire_historique_à_date(fin=fin, historique=historique))
    mci['Marque'] = mci['pdl'].isin(pdls['pdl']).apply(lambda x: 'EDN' if x else 'ZEL')
    mci = mci[mci['Marque'] == 'EDN']

    _masque = (historique['pdl'].isin(mci['pdl'])) & (historique["Date_Evenement"] >= deb) & (historique["Date_Evenement"] <= fin) & historique['Evenement_Declencheur'].isin(['CFNE', 'CFNS', 'MES', 'PES', 'RES'])
    es_mci = historique[_masque]
    es_mci
    return es_mci, historique, mci


@app.cell(hide_code=True)
def chargement_releves(flux_path, process_flux):
    from electricore.inputs.flux import lire_flux_r151

    relevés = lire_flux_r151(process_flux('R151', flux_path / 'R151'))
    return (relevés,)


@app.cell(hide_code=True)
def visualisation_donnees_metier(es_mci, historique, mci, mo, relevés):
    mo.accordion(
        {"relevés": relevés, 
         'historique': historique, 
         'Modifications chiantes': mci, 
         'Entrées/Sorties Modifs chiantes': es_mci
        }, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# CTA""")
    return


@app.cell(hide_code=True)
def calcul_energies_taxes(deb, fin, historique, relevés):
    from electricore.core.services import facturation
    factu = facturation(deb, fin, historique, relevés, inclure_jour_fin=True)
    factu
    return (factu,)


@app.cell
def _(factu):
    factu['turpe_fixe'].sum()
    return


@app.cell
def _(mo):
    mo.md(r"""## Version 2""")
    return


@app.cell
def _(historique):
    from electricore.core.services import generer_periodes_completes
    periodes_abo = generer_periodes_completes(historique=historique)
    periodes_abo
    return (periodes_abo,)


@app.cell
def _(periodes_abo):
    t2 = periodes_abo[periodes_abo['mois_annee'].isin(['avril 2025', 'mai 2025', 'juin 2025'])]
    return (t2,)


@app.cell
def _(pdls):
    pdls
    return


@app.cell
def _(pdls, t2):
    t2_edn = t2[t2['pdl'].isin(pdls['pdl'])]
    print(t2_edn['turpe_fixe'].sum())
    t2_edn
    return


@app.cell
def _(mo):
    mo.md("""# ACCISE (ex TICFE)""")
    return


@app.cell
def _(PARIS_TZ, env, pd):
    from stationreappropriation.odoo import OdooConnector

    with OdooConnector(env) as odoo:
        lines = odoo.search_read(model='account.move.line', 
                                 filters=[[('parent_state', '=', 'posted'),
                                           ('product_uom_id', '=', 'kWh')]],
                                 fields=['display_name', 'parent_state', 'date', 'quantity']
                                ).rename(columns={'date': 'date_facturation'})

    lines["date_facturation"] = pd.to_datetime(lines["date_facturation"]).dt.tz_localize(PARIS_TZ)
    # Calculer la date de consommation (date de facturation - 1 mois)
    lines["date"] = lines["date_facturation"] - pd.DateOffset(months=1)
    # lines
    return (lines,)


@app.cell
def _(deb, fin, lines):
    filtered_lines = lines[(lines["date"] >= deb) & (lines["date"] <= fin)]
    filtered_lines
    return (filtered_lines,)


@app.cell
def _(filtered_lines):
    import math
    assiete_accise = sum(filtered_lines['quantity']) / 1000
    assiete_accise_trunc = math.trunc(assiete_accise * 1000) / 1000
    assiete_accise_trunc
    return


@app.cell
def _(mo):
    mo.md("""#Interpretation ZEL""")
    return


@app.cell
def _(Path, mo):
    file_browser = mo.ui.file_browser(
        initial_path=Path('~/data/').expanduser(), multiple=True
    )
    file_browser
    return (file_browser,)


@app.cell
def _(file_browser, pd):
    zel_data = pd.read_csv(file_browser.path(index=0), decimal=',')
    zel_data['Montant'] = pd.to_numeric(zel_data['Montant'], )
    zel_data['MWh'] = zel_data['Quantité']/1000
    zel_data['taux'] = zel_data['PUHT']*1000
    zel_data
    return (zel_data,)


@app.cell
def _(zel_data):
    print(zel_data[zel_data['Elements']=='CTA']['Quantité'].sum())
    print(zel_data[zel_data['Elements']=='CTA']['Montant'].sum())
    return


@app.cell
def _(zel_data):
    accise_data = zel_data[zel_data['Elements']=='ACCISE']
    accise_data
    return (accise_data,)


@app.cell
def _(accise_data):
    accise_data[['taux', 'MWh']].groupby('taux').sum()
    return


@app.cell
def _(accise_data):
    print(accise_data['Montant'].sum())
    return


if __name__ == "__main__":
    app.run()
