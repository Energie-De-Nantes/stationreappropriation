import marimo

__generated_with = "0.10.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np

    from datetime import date
    from pathlib import Path

    from electriflux.simple_reader import process_flux

    from stationreappropriation import load_prefixed_dotenv

    env = load_prefixed_dotenv(prefix='SR_')
    flux_path = Path('~/data/flux_enedis_v2/').expanduser()
    flux_path.mkdir(parents=True, exist_ok=True)
    return (
        Path,
        date,
        env,
        flux_path,
        load_prefixed_dotenv,
        mo,
        np,
        pd,
        process_flux,
    )


@app.cell
def _(env, pd):
    from stationreappropriation.odoo import get_pdls

    pdls = get_pdls(env)
    _local = pd.DataFrame({
        'sale.order_id': [0],  # Exemple d'identifiant de commande
        'pdl': ['14295224261882']           # Exemple de PDL
    })

    # Ajouter la nouvelle ligne à la dataframe avec pd.concat
    pdls = pd.concat([pdls, _local], ignore_index=True)
    return get_pdls, pdls


@app.cell
def _(mo):
    from stationreappropriation.utils import gen_dates
    default_start, default_end = gen_dates()
    start_date_picker = mo.ui.date(value=default_start)
    end_date_picker = mo.ui.date(value=default_end)
    mo.md(
        f"Choisis la date de début {start_date_picker} et de fin {end_date_picker}"
    )
    return (
        default_end,
        default_start,
        end_date_picker,
        gen_dates,
        start_date_picker,
    )


@app.cell
def _(end_date_picker, pd, start_date_picker):
    from zoneinfo import ZoneInfo
    PARIS_TZ = ZoneInfo("Europe/Paris")
    deb = pd.to_datetime(start_date_picker.value).tz_localize(PARIS_TZ)
    fin = pd.to_datetime(end_date_picker.value).tz_localize(PARIS_TZ)
    return PARIS_TZ, ZoneInfo, deb, fin


@app.cell
def _(PARIS_TZ, deb, fin, flux_path, pd, process_flux):
    c15 = process_flux('C15', flux_path / 'C15')
    c15['Date_Evenement'] = pd.to_datetime(c15['Date_Evenement'], utc=True).dt.tz_convert(PARIS_TZ)
    c15['Date_Releve'] = pd.to_datetime(c15['Date_Releve'], utc=True).dt.tz_convert(PARIS_TZ)
    # _filtered_c15 = c15[c15['Type_Evenement']=='CONTRAT'].copy()
    # _filtered_c15 = _filtered_c15[_filtered_c15['Date_Evenement'] <= fin]

    # c15_finperiode = _filtered_c15.sort_values(by='Date_Evenement', ascending=False).drop_duplicates(subset=['pdl'], keep='first')

    _mask = (c15['Date_Evenement'] >= deb) & (c15['Date_Evenement'] <= fin)
    c15_periode = c15[_mask]

    #c15_in_period = c15_period[c15_period['Evenement_Declencheur'].isin(['MES', 'PMES', 'CFNE'])]

    # c15_out_period = c15_period[c15_period['Evenement_Declencheur'].isin(['RES', 'CFNS'])]
    return c15, c15_periode


@app.cell
def _(c15):
    c15
    return


@app.cell
def _(c15_periode):
    c15_periode[c15_periode['Evenement_Declencheur']
            .isin(['MCT'])]
    return


@app.cell
def _(c15, deb, fin):
    from stationreappropriation.moteur_metier.consos import qui_quoi_quand

    alors = qui_quoi_quand(deb, fin, c15)
    alors
    return alors, qui_quoi_quand


@app.cell
def _(flux_path, process_flux):
    from stationreappropriation.utils import get_consumption_names

    r151 = process_flux('R151', flux_path / 'R151')
    # r151['Date_Releve'] = pd.to_datetime(r151['Date_Releve'], utc=True).dt.tz_convert(PARIS_TZ)
    # Dans le r151, les index sont donnés en Wh, ce qui n'est pas le cas dans les autres flux, on va donc passer en kWh. On ne facture pas des fractions de Kwh dans tous les cas. 
    conso_cols = [c for c in get_consumption_names() if c in r151]
    r151[conso_cols] = (r151[conso_cols] / 1000).round().astype('Int64')
    r151['Unité'] = 'kWh'
    r151
    return conso_cols, get_consumption_names, r151


@app.cell
def _(alors, deb, fin, r151):
    from stationreappropriation.moteur_metier.consos import ajout_R151

    ajout_R151(deb, fin, alors, r151)
    return (ajout_R151,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
