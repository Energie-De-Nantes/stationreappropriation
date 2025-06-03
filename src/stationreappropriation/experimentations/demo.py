import marimo

__generated_with = "0.9.34"
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
def __(mo):
    from stationreappropriation.utils import gen_dates
    default_start, default_end = gen_dates()
    start_date_picker = mo.ui.date(value=default_start)
    end_date_picker = mo.ui.date(value=default_end)
    mo.md(
        f"""
        ## Délimitation temporelle
        Choisis la date de début {start_date_picker} et de fin {end_date_picker}\n
        """
    )
    return (
        default_end,
        default_start,
        end_date_picker,
        gen_dates,
        start_date_picker,
    )


@app.cell
def __(mo):
    mo.md(
        """
        # Flux Enedis
        ## Téléchargement des flux
        """
    )
    return


@app.cell(hide_code=True)
def __(env, flux_path, mo):
    from stationreappropriation.marimo_utils import download_with_marimo_progress as _dl

    _processed, _errors = _dl(env, ['R15', 'R151', 'C15', 'F15', 'F12'], flux_path)

    mo.md(f"Processed #{len(_processed)} files, with #{len(_errors)} erreurs")
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Récupérer la list de nos pds

        Alors oui, cela fait appel à Odoo. Mais ça va être utile pour différencier nos marques dans les flux.
        """
    )
    return


@app.cell
def __(env):
    from stationreappropriation.odoo import get_pdls, get_valid_subscriptions_pdl
    pdls = get_pdls(env)
    subs = get_valid_subscriptions_pdl(env)
    return get_pdls, get_valid_subscriptions_pdl, pdls, subs


@app.cell
def __(mo):
    mo.md(
        r"""
        # Lecture des flux
        ## Flux de Facturation
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        """
        ### F15

        Facturation des C5. 
        A priori pas tous les pros, à vérifier.
        """
    )
    return


@app.cell
def __(flux_path, pdls, process_flux, subs):
    f15 = process_flux('F15', flux_path / 'F15')
    f15['Marque'] = f15['pdl'].isin(pdls['pdl']).apply(lambda x: 'EDN' if x else 'EL')
    f15 = f15.merge(subs, on='pdl', how='left').rename(columns={'name': 'Abonnement'})
    f15
    return (f15,)


@app.cell
def __(mo):
    mo.md(r"""#### Sur la période uniquement""")
    return


@app.cell
def __(end_date_picker, f15, pd, start_date_picker):
    f15_periode = f15.copy()
    f15_periode['start_date'] = pd.to_datetime(start_date_picker.value)

    f15_periode['end_date'] = pd.to_datetime(end_date_picker.value)

    f15_periode = f15_periode[f15_periode['Date_Facture'] >= f15_periode['start_date']]
    f15_periode = f15_periode[f15_periode['Date_Facture'] <= f15_periode['end_date']]
    f15_periode = f15_periode.drop(columns=['start_date', 'end_date'])
    f15_periode
    return (f15_periode,)


@app.cell
def __(mo):
    mo.md(r"""#### Prestations uniquement""")
    return


@app.cell
def __(f15_periode):
    f15_periode[f15_periode['Unite'] == 'UNITE']
    return


@app.cell
def __(f15, f15_periode):
    f15_periode[(f15_periode['Unite'] == 'UNITE') & (f15['Marque']== 'EDN')]
    return


@app.cell
def __(mo):
    mo.md(r"""### F12""")
    return


@app.cell
def __(flux_path, pdls, process_flux, subs):
    f12 = process_flux('F12', flux_path / 'F12')
    f12['Marque'] = f12['pdl'].isin(pdls['pdl']).apply(lambda x: 'EDN' if x else 'EL')
    f12 = f12.merge(subs, on='pdl', how='left').rename(columns={'name': 'Abonnement'})
    f12
    return (f12,)


@app.cell
def __(end_date_picker, f12, pd, start_date_picker):
    f12_periode = f12.copy()
    f12_periode['start_date'] = pd.to_datetime(start_date_picker.value)

    f12_periode['end_date'] = pd.to_datetime(end_date_picker.value)

    f12_periode = f12_periode[f12_periode['Date_Facture'] >= f12_periode['start_date']]
    f12_periode = f12_periode[f12_periode['Date_Facture'] <= f12_periode['end_date']]
    f12_periode = f12_periode.drop(columns=['start_date', 'end_date'])
    f12_periode
    return (f12_periode,)


@app.cell
def __(f12_periode):
    f12_periode[(f12_periode['Unite'] == 'UNITE') & (f12_periode['Marque']== 'EDN')]
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        """
        ### Flux contractuel (C15)

        Le flux C15 est le flux contractuel, chaque ligne correspond à une modification sur un PRM. L'événement peut prendre les valeurs suivantes :

        | Type d'Événement | Nature d'Événement | Signification |
        |------------------|--------------------|--------------|
        | CONTRAT          | PMES               | Première mise en service |
        | CONTRAT          | MES                | Mise en service |
        | CONTRAT          | RES                | Résiliation |
        | CONTRAT          | CFNS               | Changement de fournisseur sortant |
        | CONTRAT          | CFNE               | Changement de fournisseur entrant |
        | CONTRAT          | MCT                | Modification de la formule tarifaire d'acheminement ou de la puissance souscrite ou du statut d'Autoconsommation Collective |
        | CONTRAT          | MDCTR              | Modification manuelle d'une donnée de la situation contractuelle (puissance souscrite, catégorie client, contexte d'utilisation, caractère provisoire du contrat |
        | CONTRAT          | MDACT              | Modification d'information acteur (titulaire ou interlocuteur contrat) via M007 ou modification manuelle |
        | CONTRAT          | AUTRE              | Annulation de contrat, autres modifications contractuelles liées au contrat d'injection |
        | TECHNIQUE        | MDBRA              | Modification de données de branchement |
        | TECHNIQUE        | COU                | Coupure ou limitation de puissance |
        | TECHNIQUE        | RET                | Rétablissement après coupure ou limitation de puissance |
        | TECHNIQUE        | CMAT               | Changement de compteur ou de disjoncteur ou activation du calendrier Distributeur |
        """
    )
    return


@app.cell
def __(flux_path, pdls, process_flux):
    c15 = process_flux('C15', flux_path / 'C15')
    c15['Marque'] = c15['pdl'].isin(pdls['pdl']).apply(lambda x: 'EDN' if x else 'EL')
    c15
    return (c15,)


@app.cell
def __(c15):
    c15[c15['Date_Releve'].isnull()]
    return


@app.cell
def __(mo):
    mo.md(r"""#### Situation actuelle par pdl""")
    return


@app.cell
def __(c15):
    c15_latest = c15[c15['Type_Evenement']=='CONTRAT'].sort_values(by='Date_Evenement', ascending=False).drop_duplicates(subset=['pdl'], keep='first')
    c15_latest
    return (c15_latest,)


@app.cell
def __(mo):
    mo.md(
        """
        #### Situation à date de FIN

        On va filtrer le C15 pour ne pas prendre en compte les événements contractuels post date de fin
        """
    )
    return


@app.cell
def __(c15, end_date_picker, pd):
    _filtered_c15 = c15[c15['Type_Evenement']=='CONTRAT'].copy()

    _filtered_c15['end_date'] = pd.to_datetime(end_date_picker.value)

    _filtered_c15 = _filtered_c15[_filtered_c15['Date_Evenement'] <= _filtered_c15['end_date']]

    c15_datefin = _filtered_c15.sort_values(by='Date_Evenement', ascending=False).drop_duplicates(subset=['pdl'], keep='first')
    c15_datefin
    return (c15_datefin,)


@app.cell
def __(mo):
    mo.md(r"""#### Évolutions pendant la période""")
    return


@app.cell
def __(c15, end_date_picker, pd, start_date_picker):
    _mask = (c15['Date_Evenement'] >= pd.to_datetime(start_date_picker.value)) & (c15['Date_Evenement'] <= pd.to_datetime(end_date_picker.value))
    c15_period = c15[_mask]
    c15_period
    return (c15_period,)


@app.cell
def __(mo):
    mo.md(
        """
        #### Entrées 

        On filtre les `Evenement_Declencheur`, pour n'obtenir que les entrées :

        - RES : Mise en service
        - PMES : Première mise en service
        - CFNE : Changement de fournisseur entrant
        """
    )
    return


@app.cell
def __(c15):
    c15_in = c15[c15['Evenement_Declencheur'].isin(['MES', 'PMES', 'CFNE'])]
    c15_in
    return (c15_in,)


@app.cell
def __(mo):
    mo.md(r"""#### Entrées dans la période""")
    return


@app.cell
def __(c15_period):
    c15_in_period = c15_period[c15_period['Evenement_Declencheur'].isin(['MES', 'PMES', 'CFNE'])]
    return (c15_in_period,)


@app.cell
def __(mo):
    mo.md(
        """
        #### Sorties 

        On filtre les `Evenement_Declencheur`, pour n'obtenir que les sorties :
         - RES : Résiliation
         - CFNS : Changement de fournisseur sortant
        """
    )
    return


@app.cell
def __(c15):
    c15_out = c15[c15['Evenement_Declencheur'].isin(['RES', 'CFNS'])]
    c15_out
    return (c15_out,)


@app.cell
def __(mo):
    mo.md(r"""#### Sorties dans la période""")
    return


@app.cell
def __(c15_period):
    c15_out_period = c15_period[c15_period['Evenement_Declencheur'].isin(['RES', 'CFNS'])]
    c15_out_period
    return (c15_out_period,)


@app.cell
def __(c15_period):
    c15_period[(c15_period['Evenement_Declencheur'].isin(['RES', 'CFNS'])) & (c15_period['Marque'] == 'EDN')]
    return


@app.cell
def __(mo):
    mo.md(
        """
        #### Modifications Importantes

        Je crois qu'il n'y a que MCT
        """
    )
    return


@app.cell
def __(c15):
    c15_mct = c15[c15['Evenement_Declencheur'].isin(['MCT'])]
    c15_mct
    return (c15_mct,)


@app.cell
def __(c15_period):
    c15_period_mct = c15_period[(c15_period['Evenement_Declencheur'].isin(['MCT'])) & (c15_period['Marque']=='EDN')]
    c15_period_mct
    return (c15_period_mct,)


@app.cell
def __(c15, c15_period_mct):
    c15[c15['pdl'].isin(c15_period_mct['pdl'])].sort_values(by=['pdl', 'Date_Releve'])
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ### Index

        Récupération de tous les relevés quotidiens
        """
    )
    return


@app.cell
def __(flux_path, process_flux):
    from stationreappropriation.utils import get_consumption_names

    r151 = process_flux('R151', flux_path / 'R151')

    # Dans le r151, les index sont donnés en Wh, ce qui n'est pas le cas dans les autres flux, on va donc passer en kWh. On ne facture pas des fractions de Kwh dans tous les cas. 
    conso_cols = [c for c in get_consumption_names() if c in r151]
    #r151[conso_cols] = r151[conso_cols].apply(pd.to_numeric, errors='coerce')
    r151[conso_cols] = (r151[conso_cols] / 1000).round().astype('Int64')
    r151['Unité'] = 'kWh'
    r151
    return conso_cols, get_consumption_names, r151


@app.cell
def __(mo):
    mo.md(
        """
        #### Index de départ

        On filtre tq `Date_Releve` = date départ choisie
        """
    )
    return


@app.cell
def __(pd, r151, start_date_picker):
    start_index = r151.copy()

    start_index['start_date'] = pd.to_datetime(start_date_picker.value)

    start_index = start_index[start_index['Date_Releve']==start_index['start_date']]

    start_index
    return (start_index,)


@app.cell
def __(end_date_picker, pd, r151):
    end_index = r151.copy()

    end_index['end_date'] = pd.to_datetime(end_date_picker.value)

    end_index = end_index[end_index['Date_Releve']==end_index['end_date']]
    end_index
    return (end_index,)


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Fusion
        ### Principe
        """
    )
    return


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


@app.cell
def __(mo):
    mo.md(r"""### Application""")
    return


@app.cell(hide_code=True)
def __(
    c15_datefin,
    c15_in_period,
    c15_out_period,
    conso_cols,
    end_index,
    mo,
    start_index,
):
    # Base : C15 Actuel
    _merged_enedis_data = c15_datefin[['pdl', 
                                      'Formule_Tarifaire_Acheminement', 
                                      'Puissance_Souscrite', 
                                      'Num_Depannage', 
                                      'Type_Compteur', 
                                      'Num_Compteur']]
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


    if not _duplicates_df.empty:
        _to_ouput = mo.vstack([mo.callout(mo.md(f"""
                                                **Attention: Il y a {len(_duplicates_df)} entrées dupliquées dans les données !**
                                                Pour la suite, le pdl problématique sera écarté, les duplicatas sont affichés ci-dessous."""), kind='warn'),
                               _duplicates_df.dropna(axis=1, how='all')])
    else:
        _to_ouput = mo.callout(mo.md(f'Fusion réussie'), kind='success')

    _to_ouput
    return (enedis_data,)


@app.cell
def __(enedis_data):
    enedis_data
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        """
        # Calculs des consos
        ## Choix des index

        Principe : A partir des données d'Enedis, on choisit les index à utiliser : 

        Pour l'index de début de période, on choisit une entrée (CFNE ou MES) si elle existe, sinon on utilise l'index donné par le flux d'index quotidiens (R151) du premier jour de la période.

        Pour l'index de fin de période, on choisit une sortie (CFNs ou RES) si elle existe, sinon on utilise l'index donné par le flux d'index quotidiens (R151) du dernier jour de la période.
        """
    )
    return


@app.cell
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


@app.cell
def __(mo):
    mo.md(
        """
        ## Soustraction des index

        On prend l'index de fin sélectionné précédemment, et on y soustrait l'index de début
        """
    )
    return


@app.cell(hide_code=True)
def __(DataFrame, get_consumption_names, indexes, np, pd):
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
    consos = _compute_missing_sums(consos)[['pdl', 'Formule_Tarifaire_Acheminement', 'Puissance_Souscrite', 'Num_Depannage', 'Type_Compteur', 'Num_Compteur', 'missing_data', 'd_date', 'f_date',]+_cols]

    consos['j'] = (pd.to_datetime(consos['f_date']) - pd.to_datetime(consos['d_date'])).dt.days + 1
    consos
    return (consos,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""## Calcul des taxes""")
    return


@app.cell(hide_code=True)
def __(mo, pd):
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

    cg = 15.48
    cc = 19.9
    tcta = 0.2193
    mo.md(
        f"""
        ### Turpe

        Composante de Gestion annuelle $cg = {cg}$\n
        Composante de Comptage annuelle $cc = {cc}$\n
        Cta $cta = {tcta} * turpe fixe$
        """   
    )
    return b, c, cc, cg, tcta


@app.cell(hide_code=True)
def __(b, c, mo):
    mo.vstack([
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
        ]
    )
    return


@app.cell(hide_code=True)
def __(b, c, cc, cg, consos, np, tcta):
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
    taxes
    return (taxes,)


if __name__ == "__main__":
    app.run()
