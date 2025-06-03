import marimo

__generated_with = "0.9.34"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def __():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import logging
    from datetime import date
    from pathlib import Path

    from electriflux.simple_reader import process_flux

    from stationreappropriation import load_prefixed_dotenv
    from stationreappropriation.utils import gen_previous_month_boundaries, gen_last_months

    env = load_prefixed_dotenv(prefix='SR_')
    # env['ODOO_URL'] = 'https://edn-duplicate.odoo.com/'
    # env['ODOO_DB'] = 'edn-duplicate'
    flux_path = Path('~/data/flux_enedis_v2/').expanduser()
    flux_path.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,  # Set to DEBUG to capture detailed logs
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),  # Log to console
            logging.FileHandler("marimo.log")  # Log to file (you can specify your file path)
        ]
    )
    logger = logging.getLogger(__name__)
    return (
        Path,
        date,
        env,
        flux_path,
        gen_last_months,
        gen_previous_month_boundaries,
        load_prefixed_dotenv,
        logger,
        logging,
        mo,
        np,
        pd,
        process_flux,
    )


@app.cell(hide_code=True)
def __(env, flux_path, mo):
    from stationreappropriation.marimo_utils import download_with_marimo_progress as _dl

    _processed, _errors = _dl(env, ['R15', 'R151', 'C15', 'F15', 'F12'], flux_path)

    mo.md(f"Processed #{len(_processed)} files, with #{len(_errors)} erreurs")
    return


@app.cell(hide_code=True)
def __(date, gen_previous_month_boundaries, mo):
    default_start, default_end = gen_previous_month_boundaries()
    start_date_picker = mo.ui.date(value=date(year=2024, month=1, day=1))
    end_date_picker = mo.ui.date(value=default_end)
    mo.md(
        f"""
        Tu peux aussi choisir directement les dates de début {start_date_picker} et de fin {end_date_picker}\n
        """
    )
    return default_end, default_start, end_date_picker, start_date_picker


@app.cell(hide_code=True)
def __(end_date_picker, pd, start_date_picker):
    start_time = pd.to_datetime(start_date_picker.value)
    end_time = pd.to_datetime(end_date_picker.value)
    return end_time, start_time


@app.cell
def __(taxes):
    base = taxes.copy()
    base.dropna(subset=['BASE', 'j', 'Abonnement'], inplace=True)
    base.drop(columns=['Num_Depannage', 'Num_Compteur', 'd_date', 'f_date', 'Date_Evenement', 'Evenement_Declencheur', 'missing_data', 'Date_Derniere_Modification_FTA', 'sale.order_id'], inplace=True)
    base
    return (base,)


@app.cell
def __(acci, base, capa, equi, prod):
    couts = base.copy()
    couts['var_kWh'] = (equi.value + capa.value + prod.value + acci.value) / 1000

    couts['cout_var'] = couts['var_kWh']*couts['BASE'] + couts['turpe_var']
    couts['cout_fix'] = couts['turpe_fix'] + couts['cta']
    couts['cout_total_periode'] = couts['cout_var'] + couts['cout_fix']

    couts['cout_var_proj'] = couts['var_kWh']*couts['BASE'] + couts['turpe_var_proj']
    couts['cout_fix_proj'] = couts['turpe_fix_proj'] + couts['cta']
    couts['cout_total_periode_proj'] = couts['cout_var_proj'] + couts['cout_fix_proj']
    couts
    return (couts,)


@app.cell
def __(env):
    from stationreappropriation.odoo import OdooConnector

    with OdooConnector(env) as _odoo:
        # Récupérer les lignes de facture avec la relation vers les commandes de vente
        lines = _odoo.search_read(
            model='account.move.line',
            filters=[[('parent_state', '=', 'posted'), ('product_uom_id', '=', 'kWh')]],
            fields=['display_name', 'move_id', 'price_total']
        ).rename(columns={'date': 'date_facturation'})

        # Obtenir les IDs des factures à partir de 'move_id'
        invoice_ids = lines['move_id'].apply(lambda x: x[0] if hasattr(x, '__iter__') else x).astype(int).tolist()

        # Récupérer les factures avec le champ qui pointe vers 'sale.order'
        _invoices = _odoo.read(
            model='account.move',
            ids=invoice_ids,
            fields=['id', 'invoice_origin']
        )
        lines['invoice_origin'] = _invoices['invoice_origin']

    total_var_facture = lines.groupby('invoice_origin')['price_total'].sum().reset_index().rename(columns={'price_total': 'fact_var'})
    total_var_facture
    return OdooConnector, invoice_ids, lines, total_var_facture


@app.cell
def __(OdooConnector, env):
    with OdooConnector(env) as _odoo:
        # Récupérer les lignes de facture avec la relation vers les commandes de vente
        _lines = _odoo.search_read(
            model='account.move.line',
            filters=[[('parent_state', '=', 'posted'), ('product_uom_id', '=', 'Days')]],
            fields=['display_name', 'move_id', 'price_total']
        ).rename(columns={'date': 'date_facturation'})

        # Obtenir les IDs des factures à partir de 'move_id'
        _invoice_ids = _lines['move_id'].apply(lambda x: x[0] if hasattr(x, '__iter__') else x).astype(int).tolist()

        # Récupérer les factures avec le champ qui pointe vers 'sale.order'
        _invoices = _odoo.read(
            model='account.move',
            ids=_invoice_ids,
            fields=['id', 'invoice_origin']
        )
        _lines['invoice_origin'] = _invoices['invoice_origin']
    total_fix_facture = _lines.groupby('invoice_origin')['price_total'].sum().reset_index().rename(columns={'price_total': 'fact_fix'})
    total_fix_facture
    return (total_fix_facture,)


@app.cell
def __(couts, total_fix_facture, total_var_facture, var_trv):
    marge = couts[(couts['lisse']==False) & (couts['j'] >=30)].copy()
    marge = marge.merge(total_var_facture, left_on='Abonnement', right_on='invoice_origin', how='left')
    marge = marge.merge(total_fix_facture, left_on='Abonnement', right_on='invoice_origin', how='left')
    marge.dropna(subset=['fact_var', 'fact_fix'], inplace=True)

    marge['marge_var'] = marge['fact_var'] - marge['cout_var']
    marge['marge_var_jour'] = marge['marge_var'] / marge['j']
    marge['marge_var_mois'] = marge['marge_var_jour'] * 30

    marge['marge_fix'] = marge['fact_fix'] - marge['cout_fix']
    marge['marge_fix_jour'] = marge['marge_fix'] / marge['j']
    marge['marge_fix_mois'] = marge['marge_fix_jour'] * 30

    marge['marge_mois'] = marge['marge_var_mois'] + marge['marge_fix_mois']

    marge['marge_var_proj'] = marge['fact_var']*(1+var_trv.value/100) - marge['cout_var_proj']
    marge['marge_var_jour_proj'] = marge['marge_var_proj'] / marge['j']
    marge['marge_var_mois_proj'] = marge['marge_var_jour_proj'] * 30
    marge['marge_var_kwh_proj'] = marge['marge_var_proj'] / marge["BASE"]

    marge['marge_fix_proj'] = marge['fact_fix'] - marge['cout_fix_proj']
    marge['marge_fix_jour_proj'] = marge['marge_fix_proj'] / marge['j']
    marge['marge_fix_mois_proj'] = marge['marge_fix_jour_proj'] * 30

    marge['marge_mois_proj'] = marge['marge_var_mois_proj'] + marge['marge_fix_mois_proj']

    _columns = [col for col in marge.columns if col != 'Abonnement'] + ['Abonnement']
    marge = marge[_columns]
    marge
    return (marge,)


@app.cell
def __(marge, pd):
    _data = {
        'Metric': ['Mean', 'Median'],
        'marge part fixe': [marge['marge_fix_mois_proj'].mean(), marge['marge_fix_mois_proj'].median()],
        'marge part variable': [marge['marge_var_mois_proj'].mean(), marge['marge_var_mois_proj'].median()],
        'marge par kWh': [marge['marge_var_kwh_proj'].mean(), marge['marge_var_kwh_proj'].median()],
        'marge mensuelle': [marge['marge_mois_proj'].mean(), marge['marge_mois_proj'].median()],
    }
    summary_df = pd.DataFrame(_data).round(3)

    summary_df.set_index('Metric').T
    return (summary_df,)


@app.cell(hide_code=True)
def __(marge, mo):
    import altair as alt
    # Créer un histogramme de la distribution de la marge mensuelle
    _reel = alt.Chart(marge).mark_bar().encode(
        alt.X('marge_mois', bin=alt.Bin(maxbins=20), title='Marge Mensuelle'),
        alt.Y('count()', title='Nombre de souscriptions')
    ).properties(
        title='Distribution de la Marge Mensuelle'
    )

    _proj = alt.Chart(marge).mark_bar().encode(
        alt.X('marge_mois_proj', bin=alt.Bin(maxbins=20), title='Marge Mensuelle'),
        alt.Y('count()', title='Nombre de souscriptions')
    ).properties(
        title='Distribution de la Marge Mensuelle'
    )
    _combined_chart = alt.vconcat(_reel, _proj)
    mo.ui.altair_chart(_combined_chart)
    return (alt,)


@app.cell
def __(alt, marge, mo):
    # Créer un histogramme de la distribution de la marge mensuelle avec une facette par Puissance_Souscrite
    _chart = alt.Chart(marge).mark_bar().encode(
        alt.X('marge_mois', bin=alt.Bin(maxbins=30), title='Marge Mensuelle'),
        alt.Y('count()', title='Nombre de lignes'),
        alt.Color('Puissance_Souscrite', legend=alt.Legend(title="Puissance Souscrite"))
    ).facet(
        'Puissance_Souscrite',
        columns=3
    ).properties(
        title='Distribution de la Marge Mensuelle par Puissance Souscrite'
    )
    mo.ui.altair_chart(_chart)
    return


@app.cell
def __(acci, capa, equi, mo, prod):
    mo.md(
        f"""
        # Ajout des couts
        ## Fixes :

         - CTA
         - Turpe Fixe
        ## Variables :

         - Turpe var
         - accise
         - equilibrage ({equi.value}€/MWh)
         - capacité ({capa.value}€/MWh) 
         - production ({prod.value}€/MWh) 
         - accise ({acci.value}€/MWh) 

        """
    )
    return


@app.cell
def __(mo):
    equi = mo.ui.slider(start=1, stop=3, step=.01, label="Equilibrage (€/MWh)", value=1.85)
    capa = mo.ui.slider(start=2, stop=10, step=.01, label="Capacité    (€/MWh)", value=4)
    prod = mo.ui.slider(start=50, stop=140, step=1, label="Production  (€/MWh)", value=100)
    acci = mo.ui.slider(start=15, stop=50, step=.01, label="Accise      (€/MWh)", value=33.7)
    aug_turpe = mo.ui.slider(start=-20, stop=30, step=1, label="Augmentation du Turpe (%)", value=7)
    var_trv = mo.ui.slider(start=-20, stop=10, step=1, label="Variation du TRVE (%)", value=-15)
    mo.vstack([
        equi,
        capa,
        prod,
        acci,
        aug_turpe,
        var_trv
    ])
    return acci, aug_turpe, capa, equi, prod, var_trv


@app.cell
def __(mo):
    mo.md(r"""# Détail des calculs ENEDIS""")
    return


@app.cell
def c15(end_date_picker, flux_path, pd, process_flux, start_date_picker):
    c15 = process_flux('C15', flux_path / 'C15')
    _filtered_c15 = c15[c15['Type_Evenement']=='CONTRAT'].copy()
    _filtered_c15 = _filtered_c15[_filtered_c15['Date_Evenement'] <= pd.to_datetime(end_date_picker.value)]

    c15_finperiode = _filtered_c15.sort_values(by='Date_Evenement', ascending=False).drop_duplicates(subset=['pdl'], keep='first')

    _mask = (c15['Date_Evenement'] >= pd.to_datetime(start_date_picker.value)) & (c15['Date_Evenement'] <= pd.to_datetime(end_date_picker.value))
    c15_period = c15[_mask]

    c15_in_period = c15_period[c15_period['Evenement_Declencheur'].isin(['MES', 'PMES', 'CFNE'])]

    c15_out_period = c15_period[c15_period['Evenement_Declencheur'].isin(['RES', 'CFNS'])]
    return c15, c15_finperiode, c15_in_period, c15_out_period, c15_period


@app.cell
def __(mo):
    mo.md("""## Changements dans la période""")
    return


@app.cell
def __(c15_period):
    c15_mct = c15_period[c15_period['Evenement_Declencheur'].isin(['MCT'])]
    c15_mct
    return (c15_mct,)


@app.cell
def __(end_date_picker, flux_path, pd, process_flux, start_date_picker):
    from stationreappropriation.utils import get_consumption_names

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
    return conso_cols, end_index, get_consumption_names, r151, start_index


@app.cell(hide_code=True)
def __(end_date_picker, start_date_picker):
    from stationreappropriation.graphics import plot_data_merge

    _graphique_data = [
        ('C15 (fin periode)', ['FTA', 'Puissance_Sousc.', 'Num_Depannage', 'Type_Compteur', 'Num_Compteur']),
        ('C15 (IN periode)', ['date IN', 'index IN']),
        ('C15 (OUT periode)', ['date OUT', 'index OUT']),
        ('R151', [f'index {start_date_picker.value}', f'index {end_date_picker.value}']),
    ]

    plot_data_merge(_graphique_data, 'pdl')
    return (plot_data_merge,)


@app.cell(hide_code=True)
def fusion(
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
def selection_index(
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


@app.cell(hide_code=True)
def calcul_consos(DataFrame, get_consumption_names, indexes, np, pd):
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
    consos['j'] = (pd.to_datetime(consos['f_date']) - pd.to_datetime(consos['d_date'])).dt.days + 1
    return (consos,)


@app.cell(hide_code=True)
def __(mo, np, pd):
    # Création du DataFrame avec les données du tableau
    _b = {
        "b": ["CU4", "CUST", "MU4", "MUDT", "LU", "CU4 – autoproduction collective", "MU4 – autoproduction collective"],
        "€/kVA/an": [9.36, 10.44, 11.04, 12.72, 84.96, 9.36, 11.16]
    }
    b = pd.DataFrame(_b).set_index('b')
    _c = {
        "c": [
            "CU4", "CUST", "MU4", "MUDT", "LU",
            "CU 4 - autoproduction collective, part autoproduite",
            "CU 4 - autoproduction collective, part alloproduite",
            "MU 4 - autoproduction collective, part autoproduite",
            "MU 4 - autoproduction collective, part alloproduite"
        ],
        "HPH": [
            6.96, 0, 6.39, 0, 0,
            1.72, 7.56, 1.72, 6.88
        ],
        "HCH": [
            4.76, 0, 4.43, 0, 0,
            1.34, 4.62, 1.34, 4.41
        ],
        "HPB": [
            1.48, 0, 1.46, 0, 0,
            0.81, 2.39, 0.81, 2.32
        ],
        "HCB": [
            0.92, 0, 0.91, 0, 0,
            0.39, 0.9, 0.39, 0.9
        ],
        "HP": [
            0, 0, 0, 4.68, 0,
            0, 0, 0, 0
        ],
        "HC": [
            0, 0, 0, 3.31, 0,
            0, 0, 0, 0
        ],
        "BASE": [
            0, 4.58, 0, 0, 1.15,
            0, 0, 0, 0
        ]
    }
    c = pd.DataFrame(_c).set_index('c')


    tcta = 0.2193

    # Liste des puissances
    P = [3, 6, 9, 12, 15, 18, 36]

    # Constantes cg et cc
    cg = 16.20
    cc = 20.88

    # Créer la matrice selon la formule (cg + cc + b * P) / 366
    matrice = (cg + cc + b["€/kVA/an"].values[:, np.newaxis] * P) / 366

    # Créer un DataFrame à partir de la matrice
    matrice_df = pd.DataFrame(matrice, index=b.index, columns=[f'P={p} kVA' for p in P])

    mo.vstack([
        mo.md(
            f"""
            ### Turpe

            Composante de Gestion annuelle $cg = {cg}$\n
            Composante de Comptage annuelle $cc = {cc}$\n
            Cta $cta = {tcta} * turpe fixe$
            """),
        mo.md(r"""
              ### Composante de soutirage

              \[
              CS = b \times P + \sum_{i=1}^{n} c_i \cdot E_i
              \]

              Dont part fixe $CSF = b \times P$
              Avec P = Puissance souscrite
              """),
        mo.hstack([b, c]), 
        mo.md(r"""
          ### Turpe Fixe journalier

          \[
          T_j = (cg + cc + b \times P)/366
          \]
          """),
        matrice_df,
        ]
    )
    return P, b, c, cc, cg, matrice, matrice_df, tcta


@app.cell(hide_code=True)
def calcul_taxes(aug_turpe, b, c, cc, cg, consos, env, np, tcta):
    taxes = consos.copy()
    # Calcul part fixe
    def _get_tarif(row):
        key = row['Formule_Tarifaire_Acheminement'].replace('BTINF', '')
        if key in b.index:
            return b.at[key, '€/kVA/an']
        else:
            return np.nan

    # On récupére les valeurs de b en fonction de la FTA
    taxes['b'] = taxes.apply(_get_tarif, axis=1)
    taxes['Puissance_Souscrite'] = taxes['Puissance_Souscrite'].astype(float)

    taxes['turpe_fix_j'] = (cg + cc + taxes['b'] * taxes['Puissance_Souscrite'])/366
    taxes['turpe_fix'] = taxes['turpe_fix_j'] * taxes['j']
    taxes['cta'] = tcta * taxes['turpe_fix']

    def _calc_sum_ponderated(row):
        key = row['Formule_Tarifaire_Acheminement'].replace('BTINF', '')
        if key in c.index:
            coef = c.loc[key]
            conso_cols = ['HPH', 'HCH', 'HPB', 'HCB', 'HP', 'HC', 'BASE']
            return sum(row[col] * coef[col] for col in conso_cols)/100
        else:
            print(key)
            return 0
    taxes['turpe_var'] = taxes.apply(_calc_sum_ponderated, axis=1)
    taxes['turpe'] = taxes['turpe_fix'] + taxes['turpe_var']
    taxes['turpe_var_proj'] = taxes['turpe_var'] * (1 + aug_turpe.value / 100)
    taxes['turpe_fix_proj'] = taxes['turpe_fix'] * (1 + aug_turpe.value / 100)
    taxes['turpe_proj'] = taxes['turpe_var_proj'] + taxes['turpe_fix_proj']

    from stationreappropriation.odoo import get_valid_subscriptions_pdl
    subs = get_valid_subscriptions_pdl(env)
    taxes = taxes.merge(subs, on='pdl', how='left').rename(columns={'name': 'Abonnement'})
    taxes
    return get_valid_subscriptions_pdl, subs, taxes


if __name__ == "__main__":
    app.run()
