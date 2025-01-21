import marimo

__generated_with = "0.9.34"
app = marimo.App(width="medium")


@app.cell
def __():
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
def __(env, pd):
    from stationreappropriation.odoo import get_pdls

    pdls = get_pdls(env)
    _local = pd.DataFrame({
        'sale.order_id': [0],  # Exemple d'identifiant de commande
        'pdl': ['14295224261882']           # Exemple de PDL
    })

    # Ajouter la nouvelle ligne à la dataframe avec pd.concat
    pdls = pd.concat([pdls, _local], ignore_index=True)
    return get_pdls, pdls


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""# Délimitation temporelle""")
    return


@app.cell(hide_code=True)
def __(mo):
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


@app.cell(hide_code=True)
def __(end_date_picker, pd, start_date_picker):
    from zoneinfo import ZoneInfo
    PARIS_TZ = ZoneInfo("Europe/Paris")
    start = pd.to_datetime(start_date_picker.value).tz_localize(PARIS_TZ)
    end = pd.to_datetime(end_date_picker.value).tz_localize(PARIS_TZ)
    return PARIS_TZ, ZoneInfo, end, start


@app.cell(hide_code=True)
def __(mo):
    mo.md("""# Récupération des règles de calcul du TURPE""")
    return


@app.cell
def __(end, start):
    from stationreappropriation.moteur_metier.turpe import get_applicable_rules, compute_turpe

    turpe_rules = get_applicable_rules(start, end)
    turpe_rules
    return compute_turpe, get_applicable_rules, turpe_rules


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""# Récupération des données de conso""")
    return


@app.cell(hide_code=True)
def __(end_date_picker, flux_path, pd, process_flux, start_date_picker):
    from stationreappropriation.utils import get_consumption_names

    c15 = process_flux('C15', flux_path / 'C15')
    _filtered_c15 = c15[c15['Type_Evenement']=='CONTRAT'].copy()
    _filtered_c15 = _filtered_c15[_filtered_c15['Date_Evenement'] <= pd.to_datetime(end_date_picker.value)]

    c15_finperiode = _filtered_c15.sort_values(by='Date_Evenement', ascending=False).drop_duplicates(subset=['pdl'], keep='first')

    c15_debperiode = _filtered_c15[_filtered_c15['Date_Evenement'] <= pd.to_datetime(start_date_picker.value)].sort_values(by='Date_Evenement', ascending=False).drop_duplicates(subset=['pdl'], keep='first')

    _mask = (c15['Date_Evenement'] >= pd.to_datetime(start_date_picker.value)) & (c15['Date_Evenement'] <= pd.to_datetime(end_date_picker.value))
    c15_period = c15[_mask]

    c15_in_period = c15_period[c15_period['Evenement_Declencheur'].isin(['MES', 'PMES', 'CFNE'])]

    c15_out_period = c15_period[c15_period['Evenement_Declencheur'].isin(['RES', 'CFNS'])]


    r151 = process_flux('R151', flux_path / 'R151')

    # Dans le r151, les index sont donnés en Wh, ce qui n'est pas le cas dans les autres flux, on va donc passer en kWh. On ne facture pas des fractions de Kwh dans tous les cas. 
    conso_cols = [c for c in get_consumption_names() if c in r151]
    #r151[conso_cols] = r151[conso_cols].apply(pd.to_numeric, errors='coerce')
    r151[conso_cols] = (r151[conso_cols] / 1000).round().astype('Int64')
    r151['Unité'] = 'kWh'

    start_index = r151.copy()
    start_index['start_date'] = pd.to_datetime(start_date_picker.value)
    start_index = start_index[start_index['Date_Releve']==start_index['start_date']]

    end_index = r151.copy()
    end_index['end_date'] = pd.to_datetime(end_date_picker.value)
    end_index = end_index[end_index['Date_Releve']==end_index['end_date']]


    return (
        c15,
        c15_debperiode,
        c15_finperiode,
        c15_in_period,
        c15_out_period,
        c15_period,
        conso_cols,
        end_index,
        get_consumption_names,
        r151,
        start_index,
    )


@app.cell(hide_code=True)
def __(
    c15_finperiode,
    c15_in_period,
    c15_out_period,
    conso_cols,
    end_index,
    mo,
    start_index,
):
    # Base : C15 Actuel
    _merged_enedis_data = c15_finperiode.copy()
    # [['pdl', 
    #                                   'Formule_Tarifaire_Acheminement', 
    #                                   'Puissance_Souscrite', 
    #                                   'Num_Depannage', 
    #                                   'Type_Compteur', 
    #                                   'Num_Compteur', 
    #                                   'Segment_Clientele',
    #                                   'Categorie',    
    #                                   ]]
    def _merge_with_prefix(A, B, prefix):
        return A.merge(B.add_prefix(prefix),
                       how='left', left_on='pdl', right_on=f'{prefix}pdl'
               ).drop(columns=[f'{prefix}pdl'])
    # Fusion C15 IN
    _merged_enedis_data = _merge_with_prefix(_merged_enedis_data,
                                            c15_in_period[['pdl', 'Date_Releve']+conso_cols],
                                            'in_')

    # Fusion + C15 OUT
    _merged_enedis_data = _merge_with_prefix(_merged_enedis_data,
                                            c15_out_period[['pdl', 'Date_Releve']+conso_cols],
                                            'out_')

    # Fusion + R151 (start)
    _merged_enedis_data = _merge_with_prefix(_merged_enedis_data,
                                            start_index[['pdl']+conso_cols],
                                            'start_')
    # Fusion + R151 (end)
    _merged_enedis_data = _merge_with_prefix(_merged_enedis_data,
                                            end_index[['pdl']+conso_cols],
                                            'end_')

    # Specify the column to check for duplicates
    _duplicate_column_name = 'pdl'

    # Identify duplicates
    _duplicates_df = _merged_enedis_data[_merged_enedis_data.duplicated(subset=[_duplicate_column_name], keep=False)]

    # Drop duplicates from the original DataFrame
    enedis_data = _merged_enedis_data.drop_duplicates(subset=[_duplicate_column_name]).copy()

    erreurs = {}
    if not _duplicates_df.empty:
        _to_ouput = mo.vstack([mo.callout(mo.md(f"""
                                                **Attention: Il y a {len(_duplicates_df)} entrées dupliquées dans les données !**
                                                Pour la suite, le pdl problématique sera écarté, les duplicatas sont affichés ci-dessous."""), kind='warn'),
                               _duplicates_df.dropna(axis=1, how='all')])
        erreurs['Entrées dupliquées'] = _duplicates_df

    else:
        _to_ouput = mo.callout(mo.md(f'Fusion réussie'), kind='success')

    _to_ouput
    return enedis_data, erreurs


@app.cell(hide_code=True)
def __(
    end_date_picker,
    enedis_data,
    get_consumption_names,
    np,
    pd,
    start_date_picker,
):
    _cols = get_consumption_names()
    indexes = enedis_data.copy()
    for _col in _cols:
        indexes[f'd_{_col}'] = np.where(indexes['in_Date_Releve'].notna(),
                                                  indexes[f'in_{_col}'],
                                                  indexes[f'start_{_col}'])

    for _col in _cols:
        indexes[f'f_{_col}'] = np.where(indexes['out_Date_Releve'].notna(),
                                                  indexes[f'out_{_col}'],
                                                  indexes[f'end_{_col}'])

    indexes['start_date'] = start_date_picker.value
    indexes['start_date'] = pd.to_datetime(indexes['start_date'])#.dt.date

    indexes['end_date'] = end_date_picker.value
    indexes['end_date'] = pd.to_datetime(indexes['end_date'])#.dt.date

    indexes[f'd_date'] = np.where(indexes['in_Date_Releve'].notna(),
                                         indexes[f'in_Date_Releve'],
                                         indexes[f'start_date'])
    indexes[f'f_date'] = np.where(indexes['out_Date_Releve'].notna(),
                                         indexes[f'out_Date_Releve'],
                                         indexes[f'end_date'])

    indexes[f'd_date'] = pd.to_datetime(indexes['d_date'])
    indexes[f'f_date'] = pd.to_datetime(indexes['f_date'])
    return (indexes,)


app._unparsable_cell(
    r"""
    _cols = get_consumption_names()
    consos = indexes.copy()

    # Calcul des consommations
    for _col in _cols:
        consos[f'{_col}'] = consos[f'f_{_col}'] - consos[f'd_{_col}']

    def _compute_missing_sums(df: DataFrame) -> DataFrame:
        if 'BASE' not in df.columns:
            df['BASE'] = np.nan  

        df['missing_data'] = df[['HPH', 'HPB', 'HCH', 
                'HCB', 'BASE', 'HP',
                'HC']].isna().all(axis=1)
        df['BASE'] = np.where(
                df['missing_data'],
                np.nan,
                df[['HPH', 'HPB', 'HCH', 
                'HCB', 'BASE', 'HP', 
                'HC']].sum(axis=1)
            )
        df['HP'] = df[['HPH', 'HPB', 'HP']].sum(axis=1)
        df['HC'] = df[['HCH', 'HCB', 'HC']].sum(axis=1)
        return df.copy()
    consos = _compute_missing_sums(consos)
    consos = consos[['pdl', 
                     'Formule_Tarifaire_Acheminement',
                     'Puissance_Souscrite',
                     'Num_Depannage',
                     'Type_Compteur',
                     'Num_Compteur',
                     'missing_data',
                     'd_date',
                     'Segment_Clientele',
                     'Categorie',
                     'Date_Derniere_Modification_FTA',
                     'Etat_Contractuel',
                     'Evenement_Declencheur',
                     'Date_Evenement',
                     'f_date',]+_cols]
    consos['j'] = (pd.to_datetime(consos['f_date']) - pd.to_datetime(consos['d_date'])).dt.days.clip(lower=0)
    consos['FTA'] = 
    """,
    name="__",
    column=None, disabled=False, hide_code=True
)


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""# Détection des MCT""")
    return


@app.cell
def __(c15_period, consos, np, pd, pdls):
    c15_mct = c15_period[c15_period['Evenement_Declencheur'].isin(['MCT'])]
    c15_mct['Marque'] = c15_mct['pdl'].isin(pdls['pdl']).apply(lambda x: 'EDN' if x else 'ZEL')
    safe = consos[~consos["pdl"].isin(c15_mct["pdl"])].copy()
    safe = safe.rename(columns={'d_date': 'start', 'f_date': 'end'})
    safe['Puissance_Souscrite'] = pd.to_numeric(safe['Puissance_Souscrite'])
    # Select only numeric columns
    numeric_cols = safe.select_dtypes(include=[np.number]).columns

    # Use .clip(lower=0) to replace negative values with 0
    safe[numeric_cols] = safe[numeric_cols].clip(lower=0)
    c15_mct
    return c15_mct, numeric_cols, safe


@app.cell
def __(PARIS_TZ, c15, c15_mct, end, np, pd, pdls):
    _cond = [
        c15['pdl'].isin(c15_mct['pdl']),
        pd.to_datetime(c15['Date_Evenement']).dt.tz_localize(PARIS_TZ) <= end,
    ]
    histo = c15[np.logical_and.reduce(_cond)].copy()
    histo['Marque'] = histo['pdl'].isin(pdls['pdl']).apply(lambda x: 'EDN' if x else 'ZEL')
    histo
    return (histo,)


@app.cell
def __(safe):
    safe
    return


@app.cell
def __(turpe_rules):
    turpe_rules
    return


@app.cell
def __(compute_turpe, pdls, safe, turpe_rules):
    turpe = compute_turpe(safe, turpe_rules)
    turpe['Marque'] = turpe['pdl'].isin(pdls['pdl']).apply(lambda x: 'EDN' if x else 'ZEL')
    turpe
    return (turpe,)


@app.cell
def __(turpe):
    grouped = turpe[['Marque', 'turpe_fixe', 'turpe_var']].groupby('Marque').sum().round(0)
    grouped.loc['Total'] = grouped.sum(numeric_only=True)
    grouped
    return (grouped,)


if __name__ == "__main__":
    app.run()
