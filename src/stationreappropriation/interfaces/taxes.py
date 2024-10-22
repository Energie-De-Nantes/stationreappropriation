import marimo

__generated_with = "0.9.10"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
def __(env, flux_path, mo):
    from stationreappropriation.marimo_utils import download_with_marimo_progress as _dl

    _processed, _errors = _dl(env, ['R15', 'R151', 'C15', 'F15', 'F12'], flux_path)

    mo.md(f"Processed #{len(_processed)} files, with #{len(_errors)} erreurs")
    return


@app.cell(hide_code=True)
def __(mo):
    options = {"T1": 1, "T2": 2, "T3": 3, "T4": 4}
    radio = mo.ui.radio(options=options, label='Choisi le Trimestre', value="T1")
    radio
    return options, radio


@app.cell(hide_code=True)
def __(mo, radio):
    from stationreappropriation.utils import gen_trimester_dates
    _default_start, _default_end = gen_trimester_dates(radio.value)
    start_date_picker = mo.ui.date(value=_default_start)
    end_date_picker = mo.ui.date(value=_default_end)
    mo.md(
        f"""
        Date de début {start_date_picker} et de fin {end_date_picker}\n
        """
    )
    return end_date_picker, gen_trimester_dates, start_date_picker


@app.cell(hide_code=True)
def __(end_date_picker, pd, start_date_picker):
    start_time = pd.to_datetime(start_date_picker.value)
    end_time = pd.to_datetime(end_date_picker.value)
    return end_time, start_time


@app.cell
def __(mo):
    mo.md("""# Résultats""")
    return


@app.cell(hide_code=True)
def __(np, taxes):
    conditions = [
        taxes['Date_Derniere_Modification_FTA'] >= taxes['d_date'],
        taxes['Date_Derniere_Modification_FTA'] <= taxes['f_date'],
    ]

    conditions = [
        taxes['Date_Evenement'] >= taxes['d_date'],
        taxes['Date_Evenement'] <= taxes['f_date'],
        taxes['Evenement_Declencheur'].isin(['MCT', 'MDCTR'])
    ]

    _mask = np.logical_and.reduce(conditions)
    changements_impactants = taxes[_mask]
    changements_impactants
    return changements_impactants, conditions


@app.cell
def __(c15, changements_impactants, end_time, np):
    _cond = [
        c15['pdl'].isin(changements_impactants['pdl']),
        c15['Date_Evenement'] <= end_time,
    ]
    c15[np.logical_and.reduce(_cond)]
    return


@app.cell
def __(mo):
    mo.md(r"""Note pour plus tard, dans le cas ou l'on a une MCT, on peut diviser la ligne du pdl en deux lignes, avec des puissances et ou FTA spécifiques, ainsi que le nb de jours associés à cette configuration.""")
    return


@app.cell
def __(c15_finperiode):
    c15_finperiode
    return


@app.cell(hide_code=True)
def __(taxes):
    grouped_df = taxes.groupby('Marque')['turpe_fix'].sum().round().reset_index()
    grouped_df
    return (grouped_df,)


@app.cell(hide_code=True)
def __(accise, assiete_accise_trunc, erreurs, mo, taxes, tcta):
    assiette_cta = round(sum(taxes['turpe_fix']))
    cta = round(assiette_cta*tcta)
    mo.callout(mo.md(f"""Assiette CTA : **{assiette_cta}€**\n
                   CTA : **{cta}€**\n
                   Assiete accise : **{assiete_accise_trunc}MWh**\n
                   accise : **{accise}€**
                """),kind='success' if not erreurs else 'danger')
    return assiette_cta, cta


@app.cell(hide_code=True)
def __(erreurs, mo, taxes):
    mo.vstack([mo.accordion(erreurs),
               mo.md('Tableau calcul CTA'),
               taxes,
              ])
    return


@app.cell
def __(mo):
    mo.md(r"""# Détail des calculs""")
    return


@app.cell
def __(mo):
    mo.md(r"""## CTA""")
    return


@app.cell
def __(end_date_picker, flux_path, pd, process_flux, start_date_picker):
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
        "€/kVA/an": [9.00, 9.96, 10.56, 12.24, 81.24, 9.00, 10.68]
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
            6.67, 0, 6.12, 0, 0,
            1.64, 7.23, 1.64, 6.60
        ],
        "HCH": [
            4.56, 0, 4.24, 0, 0,
            1.29, 4.42, 1.29, 4.23
        ],
        "HPB": [
            1.43, 0, 1.39, 0, 0,
            0.77, 2.29, 0.77, 2.22
        ],
        "HCB": [
            0.88, 0, 0.87, 0, 0,
            0.37, 0.86, 0.37, 0.86
        ],
        "HP": [
            0, 0, 0, 4.47, 0,
            0, 0, 0, 0
        ],
        "HC": [
            0, 0, 0, 3.16, 0,
            0, 0, 0, 0
        ],
        "BASE": [
            0, 4.37, 0, 0, 1.10,
            0, 0, 0, 0
        ]
    }
    c = pd.DataFrame(_c).set_index('c')


    tcta = 0.2193

    # Liste des puissances
    P = [3, 6, 9, 12, 15, 18, 36]

    # Constantes cg et cc
    cg = 15.48
    cc = 19.9

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
def calcul_taxes(b, c, cc, cg, consos, env, np, tcta):
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

    from stationreappropriation.odoo import get_pdls

    pdls = get_pdls(env)
    taxes['Marque'] = taxes['pdl'].isin(pdls['pdl']).apply(lambda x: 'EDN' if x else 'EL')
    taxes
    return get_pdls, pdls, taxes


@app.cell
def __(mo):
    mo.md(
        """
        L'assiette en taxes représente la base sur laquelle un impôt ou une taxe est calculé. C'est la valeur ou la quantité (comme le revenu, la valeur d'un bien, ou une quantité de consommation) sur laquelle s'applique le taux de la taxe pour déterminer le montant dû. 

        Pour la CTA, l'assiette est la part fixe du TURPE.
        """
    )
    return


@app.cell
def __(mo):
    mo.md(f"""
    Les taux de la CTA sont fixés par arrêté et constituent un pourcentage de la part fixe hors taxe
    du tarif d’utilisation des réseaux de transport et de distribution d’électricité (TURPE).

    L'assiette en taxes représente la base sur laquelle un impôt ou une taxe est calculé. C'est la valeur ou la quantité (comme le revenu, la valeur d'un bien, ou une quantité de consommation) sur laquelle s'applique le taux de la taxe pour déterminer le montant dû. 

    Pour la CTA, l'assiette est la part fixe du TURPE. Dans notre cas, on la calcule en sommant les parts fixes du turpe de chacun des PRM/pdl.
    """)
    return


@app.cell
def __(mo):
    mo.md(
        """
        # TICFE

        [https://www.impots.gouv.fr/taxes-interieures-de-consommation-tic](https://www.impots.gouv.fr/taxes-interieures-de-consommation-tic)

        En gros, on prend les MWh facturés, a la décimale qui correspond au kWh et on multiplie par 21.

        ## Arrondis fiscaux :
        Le montant total est arrondi à l'entier le plus proche

        ## LES ARRONDIS DÉCLARATIFS
        Les données portées dans les colonnes (A) « quantités » sont exprimées en mégawattheures et sont arrondies à l’unité sans décimale
        pour la TICGN (accise sur les gaz naturels) et la TICC (accise sur les charbons).
        Pour la TICFE (accise sur l’électricité), les quantités sont exprimées en fraction de MWh à 3 décimales soit l’équivalent du KWh (0,001 MWh).
        Les données portées dans les colonnes tarifaires (B) sont toutes exprimées en € par mégawattheure (€/MWh).
        Les données portées dans les colonnes « Montant A x B » sont arrondies à 2 décimales au centime d’€ à l’exception des lignes detotalisation qui sont arrondies à l’€.
        ## CADRE 1 : TICFE (Accise sur l’électricité)
        La taxe s’applique à l'électricité reprise au code NC 27161, quelle que soit la puissance souscrite.
        """
    )
    return


@app.cell
def __(env, pd):
    from stationreappropriation.odoo import OdooConnector

    with OdooConnector(env) as odoo:
        lines = odoo.search_read(model='account.move.line', 
                                 filters=[[('parent_state', '=', 'posted'),
                                           ('product_uom_id', '=', 'kWh')]],
                                 fields=['display_name', 'parent_state', 'date', 'quantity']
                                ).rename(columns={'date': 'date_facturation'})

    lines["date_facturation"] = pd.to_datetime(lines["date_facturation"])
    # Calculer la date de consommation (date de facturation - 1 mois)
    lines["date"] = lines["date_facturation"] - pd.DateOffset(months=1)
    lines
    return OdooConnector, lines, odoo


@app.cell
def __(end_time, lines, start_time):
    # Filtrer les lignes qui sont dans le trimestre
    filtered_data = lines[(lines["date"] >= start_time) & (lines["date"] <= end_time)]
    filtered_data
    return (filtered_data,)


@app.cell
def __(filtered_data):
    import math
    assiete_accise = sum(filtered_data['quantity']) / 1000
    assiete_accise_trunc = math.trunc(assiete_accise * 1000) / 1000
    accise = assiete_accise_trunc * 21
    assiete_accise_trunc, accise
    return accise, assiete_accise, assiete_accise_trunc, math


@app.cell
def __(mo):
    mo.md(r"""Note pour plus tard, ici, on ne prend pas en compte les pro/pas pros, Attention, il faudra une méthode pour les distinguer plus tard.""")
    return


if __name__ == "__main__":
    app.run()
