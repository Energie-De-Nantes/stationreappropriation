import marimo

__generated_with = "0.9.30"
app = marimo.App(width="medium")


@app.cell
def __():
    from stationreappropriation.utils import gen_dates
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
        gen_dates,
        load_prefixed_dotenv,
        mo,
        np,
        pd,
        process_flux,
    )


@app.cell
def __(env, flux_path, mo):
    from stationreappropriation.marimo_utils import download_with_marimo_progress as _dl

    _processed, _errors = _dl(env, ['R15', 'R151', 'C15', 'F15', 'F12'], flux_path)

    mo.md(f"Téléchargement des Flux : #{len(_processed)} zip, #{len(_errors)} erreurs")
    return


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
    mo.md(
        r"""
        # Détermination de notre périmètre

        L'objectif est d'êtres capables, pour un couple `(timestamp, pdl)` donné de dire si le pdl est sur notre périmètre fournisseur. 

        Notre périmètre fournisseur comprend toutes les marques EDN, ZEL...

        ## Pourquoi ? 

        Cela nous permettra de savoir si l'on doit prendre en compte chaque ligne de la timeseries issue du M023

        ## Stratégie 

        Partir du C15, le flux contractuel, pour déterminer quand un pdl est rentré ou sorti. 
        PB, certains pdls peuvent rentrer et sortir plusieurs fois. 
        première étape, détecter un tel cas
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""C15, avec ajout de la Marque (EDN/ZEL)""")
    return


@app.cell
def __(flux_path, pdls, process_flux):
    c15 = process_flux('C15', flux_path / 'C15')
    c15['Marque'] = c15['pdl'].isin(pdls['pdl']).apply(lambda x: 'EDN' if x else 'ZEL')
    # c15 = c15[['pdl', 'Etat_Contractuel', 'Evenement_Declencheur', 'Date_Evenement', 'Marque', '']]
    c15 = c15.sort_values(by=['pdl', 'Date_Evenement'])
    c15
    return (c15,)


@app.cell
def __(c15):
    c15['pdl'].drop_duplicates().to_csv('~/data/pdl.csv', index=False, header=False)
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        A partir du C15, on construit une table des périodes dans le périmètre.  

        Chaque ligne représente un couple (pdl, période). un pdl peut avoir plusieurs périodes distinctes dans notre périmètre, en cas de déménagement puis re-souscription par exemple.
        """
    )
    return


@app.cell
def __(c15, pd):
    # Créer une liste pour stocker les périodes de validité
    _periods = []

    # Construire les périodes de validité pour chaque PDL
    for _pdl, _group in c15.groupby('pdl'):
        _in_perimeter = False
        _start_date = None

        for _, _row in _group.iterrows():
            if _row['Evenement_Declencheur'] in ['CFNE', 'MES', 'PMES']:
                # Entrée dans le périmètre
                _in_perimeter = True
                _start_date = _row['Date_Evenement']
            elif _row['Evenement_Declencheur'] in ['RES', 'CFNS'] and _in_perimeter:
                # Sortie du périmètre
                _periods.append({'pdl': _pdl, 'start': _start_date, 'end': _row['Date_Evenement']})
                _in_perimeter = False
        # Si nous sommes encore dans une période ouverte, fermer avec la date d'aujourd'hui
        if _in_perimeter:
            _periods.append({'pdl': _pdl, 'start': _start_date, 'end': pd.Timestamp.now()})
    # Convertir les périodes en DataFrame
    periods_df = pd.DataFrame(_periods)
    periods_df
    return (periods_df,)


@app.cell
def __(mo):
    mo.md(
        r"""
        # Lecture du M023 Energies Quotidiennes

        Composé de plusieurs CSVs, il indique, pour chaque jour et chaque PRM/pdl une quantité d'énergie en **Wh**.

        La méthode de calcul indiquée est *DIFF.INDEX*
        """
    )
    return


@app.cell
def __(mo):
    mo.md(r"""## Chargement""")
    return


@app.cell(hide_code=True)
def __(mo):
    file_browser = mo.ui.file_browser(
        initial_path="~/data/M023", multiple=True, filetypes=['.csv', '.json']
    )
    file_browser
    return (file_browser,)


app._unparsable_cell(
    r"""
    Après lecture du m023, on ajoute la colonne Actif, qui est construite à partir de la table de présence dans notre périmètre décrite au dessus.

    Timeserie obtenue :
    """,
    name="__"
)


@app.cell(hide_code=True)
def __(file_browser, pd, pdls, periods_df):
    _timeserie_df_list = []
    for _file in file_browser.value:
        _timeserie_df_list.append(pd.read_csv(_file.path, sep=';', parse_dates=['Date']))

    # Concaténer tous les dataframes de timeserie
    if _timeserie_df_list:
        timeserie_df = pd.concat(_timeserie_df_list, ignore_index=True)
    else:
        timeserie_df = pd.DataFrame()

    if not timeserie_df.empty:
        # Ajout de la Marque
        timeserie_df['pdl'] = timeserie_df['Identifiant PRM'].astype(str).str.zfill(14)
        timeserie_df['Marque'] = timeserie_df['pdl'].isin(pdls['pdl']).apply(lambda x: 'EDN' if x else 'ZEL')

        # Ajout perimetre
        def is_active(row, _periods_df):
            pdl_periods = _periods_df[_periods_df['pdl'] == row['pdl']]
            for _, _period in pdl_periods.iterrows():
                if (_period['start'] <= row['Date']) & (row['Date'] <= _period['end']):
                    return True
            return False


        timeserie_df['Actif'] = timeserie_df.apply(is_active, _periods_df=periods_df, axis=1)
    timeserie_df
    return is_active, timeserie_df


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## Quels ensembles sont représentés par le M023 ?

        Stratégie, comparer les pdls dans le M023 à d'autres ensembles : 

        - les pdls du périmètre (uniques dans C15)
        - les pdls linky du périmetre (Type_Compteur == CCB)
        - les pdls non linky
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Récupération du C15

        On lit puis on filtre le c15 pour ne pas prendre en compte les événements postérieurs à la timeserie :
        """
    )
    return


@app.cell
def __(c15, timeserie_df):
    max_date = timeserie_df['Date'].max()
    c15_a_date = c15[c15['Date_Evenement'] < max_date].copy()
    c15_a_date
    return c15_a_date, max_date


@app.cell
def __(mo):
    mo.md(r"""## Identification des non-linky""")
    return


@app.cell(hide_code=True)
def __(c15_a_date, mo):
    linky_mask = c15_a_date['Type_Compteur'] == 'CCB'
    linky = c15_a_date[linky_mask]['pdl'].unique()
    nlinky = c15_a_date[~linky_mask]['pdl'].unique()
    c15_pdls = c15_a_date['pdl'].unique()
    linky, nlinky
    # Tous les ZEL == linky ? 
    non_linky_zel = c15_a_date[(c15_a_date['Marque'] == 'ZEL') & (c15_a_date['Type_Compteur'] != 'CCB')]
    non_linky_edn = c15_a_date[(c15_a_date['Marque'] == 'EDN') & (c15_a_date['Type_Compteur'] != 'CCB')]
    linky_edn_pdl = c15_a_date[(c15_a_date['Marque'] == 'EDN') & (c15_a_date['Type_Compteur'] == 'CCB')]['pdl'].unique()
    non_linky_edn_pdl = non_linky_edn['pdl'].unique()
    mo.accordion({
        'ZEL non-linky': non_linky_zel,
        'EDN non-linky': non_linky_edn,
    })
    return (
        c15_pdls,
        linky,
        linky_edn_pdl,
        linky_mask,
        nlinky,
        non_linky_edn,
        non_linky_edn_pdl,
        non_linky_zel,
    )


@app.cell
def __(timeserie_df):
    m023_pdls = timeserie_df['pdl'].unique()
    return (m023_pdls,)


@app.cell
def __(mo):
    mo.md(r"""Eléments manquants dans le M023 :""")
    return


@app.cell
def __(c15_a_date, m023_pdls):
    print(f'Taille m023: {len(set(m023_pdls))}')
    print(f'Taille c15 : {len(set(c15_a_date['pdl'].unique()))}')
    print(f'Manquants dans m023 : {len(set(c15_a_date['pdl'].unique())-set(m023_pdls))}')
    # compteurs = {
    #     "CCB": "Compteur Linky",
    #     "CEB": "Compteur Electronique",
    #     "CFB": "Compteur Electromécanique"
    # }
    # c15_a_date["Type_Compteur"] = c15_a_date["Type_Compteur"].replace(compteurs)
    c15_a_date[c15_a_date['pdl'].isin(set(c15_a_date['pdl'].unique())-set(m023_pdls))]
    return


@app.cell
def __(timeserie_df):
    # Grouper par Date et par Marque et faire la somme des valeurs
    grouped_df = timeserie_df[timeserie_df['Actif']].groupby(['Date', 'Marque']).agg({'Valeur': 'sum'}).reset_index()
    return (grouped_df,)


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Consolidation des données EDN

        ### Approche simple 
        On calcule le ratio d'augmentation de la population d'EDN si on ajoute le non-linky : 

        $x = \frac{\text{linky edn} + \text{non linky edn}}{\text{linky edn}}$

        Avec l'hypothèse que chaque non-linky consomme comme la moyenne des linky, on peut faire :

        $\text{conso edn consolidée} = \text{conso edn} \times x$
        """
    )
    return


@app.cell
def __(linky_edn_pdl, mo, non_linky_edn, timeserie_df):
    piv_grouped_df = timeserie_df[timeserie_df['Actif']].pivot_table(index='Date', columns='Marque', values='Valeur', aggfunc='sum').fillna(0)
    x = (len(linky_edn_pdl) + len(non_linky_edn)) / len(linky_edn_pdl)
    piv_grouped_df['EDN_consolidee'] = piv_grouped_df['EDN']*x
    piv_grouped_df['part EDN'] = piv_grouped_df['EDN'] / (piv_grouped_df['EDN'] + piv_grouped_df['ZEL'])
    piv_grouped_df['part EDN_consolidee'] = piv_grouped_df['EDN_consolidee'] / (piv_grouped_df['EDN_consolidee'] + piv_grouped_df['ZEL'])
    mo.md(f'Ici, x = {x:.3f}')
    return piv_grouped_df, x


@app.cell(hide_code=True)
def __(mo, pd, timeserie_df, x):
    import altair as alt
    # Créer un graphique Altair des consommations produites par grouped_df
    consos_df = timeserie_df[timeserie_df['Actif']].groupby(['Date', 'Marque']).agg({'Valeur': 'sum'}).reset_index()

    # Filtrer uniquement les lignes où Marque == 'EDN'
    edn_rows = consos_df[consos_df['Marque'] == 'EDN'].copy()

    # Ajouter la nouvelle Marque "EDN_consolidé" et multiplier la valeur par x
    edn_rows['Marque'] = 'EDN_consolidé'
    edn_rows['Valeur'] *= x

    # Ajouter les nouvelles lignes consolidées à la dataframe d'origine
    consos_df = pd.concat([consos_df, edn_rows], ignore_index=True)

    conso_j_chart = alt.Chart(consos_df).mark_line().encode(
        x='Date:T',
        y='Valeur:Q',
        color='Marque:N'
    ).properties(
        title='Consommations par Marque au cours du temps'
    )
    mo.ui.altair_chart(conso_j_chart)
    return alt, conso_j_chart, consos_df, edn_rows


@app.cell
def __(mo):
    mo.md(r"""## Table des consommations totales par mois""")
    return


@app.cell
def __(piv_grouped_df, x):
    # Grouper par mois et sommer les résultats précédents
    grouped_monthly_df = piv_grouped_df.resample('ME').sum() / 1e6
    grouped_monthly_df['total'] = grouped_monthly_df['EDN'] + grouped_monthly_df['ZEL']


    grouped_monthly_df['EDN_consolide'] = grouped_monthly_df['EDN'] * x
    grouped_monthly_df['total_consolide'] = grouped_monthly_df['EDN_consolide'] + grouped_monthly_df['ZEL']
    grouped_monthly_df = grouped_monthly_df.round(3)
    grouped_monthly_df
    return (grouped_monthly_df,)


@app.cell
def __(alt, conso_j_chart, mo, pd, piv_grouped_df):
    # Reset the index to make 'Date' a column
    _piv_grouped_df = piv_grouped_df.round(2).reset_index()

    # Ensure 'Date' is in the correct format for Altair
    _piv_grouped_df['Date'] = pd.to_datetime(_piv_grouped_df['Date'])

    # Créer un graphique avec Altair
    part_edn_chart = alt.Chart(_piv_grouped_df).mark_line().encode(
        x='Date:T',  # Axe X basé sur la colonne Date (type temporel)
        y='part EDN_consolidee:Q',  # Axe Y basé sur la colonne de la part EDN consolidée (type quantitatif)
        tooltip=['Date', 'part EDN_consolidee']  # Infobulles pour des informations détaillées
    ).properties(
        title="Évolution de la part EDN consolidée",
        width=800,
        height=200
    )
    _stacked_chart = alt.vconcat(conso_j_chart, part_edn_chart).resolve_scale(
        x='shared'  # Partager l'échelle de l'axe X entre les graphiques
    )
    mo.ui.altair_chart(_stacked_chart)
    return (part_edn_chart,)


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Questions ouvertes :

        ### 1) Les consos des parcs sont-elles dans le M023 ?
        ### 2) Les non-communicants ne sont pas dedans, comment on les gère ?
        """
    )
    return


@app.cell
def __(mo):
    mo.md(r"""# Détermination des profils""")
    return


@app.cell
def __(mo):
    mo.md(r"""## Lecture S518 en provenance de axpo""")
    return


@app.cell
def __(mo):
    file_browser_s518 = mo.ui.file_browser(
        initial_path="~/data/flux_axpo/S518", multiple=True, filetypes=['.csv', '.json']
    )
    file_browser_s518
    return (file_browser_s518,)


@app.cell
def __(file_browser_s518, pd, timeserie_df):
    _s518_df_list = []
    for _file in file_browser_s518.value:
        _s518_df_list.append(pd.read_csv(_file.path, sep=';', skiprows=1, header=None, dtype=str))

    _corresp = {1: 'pdl',
                4: 'code_profil',
                13: 'start_date',
                14: 'end_date'
               }
    # Concaténer tous les dataframes de timeserie
    if _s518_df_list:
        s518_df = pd.concat(_s518_df_list, ignore_index=True).rename(columns=_corresp)
        s518_df[['profil', 'sens', 'sous_profil']] = s518_df['code_profil'].str.split('_', expand=True)
    else:
        s518_df = pd.DataFrame()

    if not timeserie_df.empty:
        ...
    s518_df
    return (s518_df,)


@app.cell
def __(pd, s518_df):
    periodes_profil_pdl = s518_df.copy()
    # Exemple : s518_df a les colonnes ['pdl', 'profil', 'start_date', 'end_date', ...]
    periodes_profil_pdl = periodes_profil_pdl.sort_values(['pdl', 'start_date'])

    periodes_profil_pdl['start_date'] = pd.to_datetime(periodes_profil_pdl['start_date'])
    periodes_profil_pdl['end_date'] = pd.to_datetime(periodes_profil_pdl['end_date'])
    # Créer l'IntervalIndex pour les dates
    intervals = pd.IntervalIndex.from_arrays(periodes_profil_pdl['start_date'], periodes_profil_pdl['end_date'], closed='both')

    # Créer un MultiIndex (pdl, interval)
    multi_idx = pd.MultiIndex.from_arrays([periodes_profil_pdl['pdl'], intervals], names=['pdl', 'interval'])

    # Réassigner l'index, puis enlever les colonnes originales de pdl, start_date, end_date
    periodes_profil_pdl = periodes_profil_pdl.set_index(multi_idx)
    periodes_profil_pdl = periodes_profil_pdl.drop(columns=['pdl'])#, 'start_date', 'end_date'])
    periodes_profil_pdl
    return intervals, multi_idx, periodes_profil_pdl


@app.cell
def __(pd, periodes_profil_pdl):
    # Exemple de récupération de profil
    p = '14295658381030'
    d = pd.Timestamp('2024-10-31')

    # Récupérer toutes les lignes pour ce pdl
    sub_df = periodes_profil_pdl.xs(p, level='pdl')

    # Vérifier quels intervalles contiennent la date d
    mask = sub_df.index.contains(d)

    if any(mask):
        profil_found = sub_df[mask]['profil'].iloc[0]
        print(f"Le profil pour {p} à la date {d} est {profil_found}")
    else:
        print(f"Aucun profil trouvé pour {p} à la date {d}.")
    return d, mask, p, profil_found, sub_df


@app.cell
def __():
    return


@app.cell
def __(periodes_profil_pdl, timeserie_df):
    def find_profil(row):
        p = row['pdl']
        d = row['Date']  # La date provient maintenant de la colonne 'Date'
        
        # On essaie de récupérer le sous-DataFrame pour le pdl spécifié
        try:
            sub_df = periodes_profil_pdl.xs(p, level='pdl')
        except KeyError:
            # Si le pdl n'existe pas dans periodes_profil_pdl, on renvoie None
            return None
        
        # sub_df a un IntervalIndex, on vérifie quel intervalle contient la date d
        mask = sub_df.index.contains(d)
        if mask.any():
            return sub_df[mask]['profil'].iloc[0]
        else:
            return None

    profiled_df = timeserie_df.copy()
    # Appliquer la fonction à chaque ligne de timeserie_df
    profiled_df['profil'] = profiled_df.apply(find_profil, axis=1)
    profiled_df
    return find_profil, profiled_df


@app.cell
def __(periodes_profil_pdl, timeserie_df):
    # PDL qui sont dans timeserie, mais pas dans periodes_profil_pdl
    set(timeserie_df['pdl']) - set(periodes_profil_pdl.index.get_level_values('pdl'))
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ### Conclusion S518 :
        Il semble y avoir pleins de trous dans la raquette temporelle. Il y a plein de périodes ou le profil n'est pas défini. Possible interpoler, mais il semble préférable d'explorer d'autres solutions en premier. 
        """
    )
    return


@app.cell
def __(mo):
    mo.md(r"""## Lecture S507 en provenance de axpo""")
    return


@app.cell
def __(mo):
    file_browser_s507 = mo.ui.file_browser(
        initial_path="~/data/flux_axpo/S507", multiple=True, filetypes=['.csv', '.json', '.txt']
    )
    file_browser_s507
    return (file_browser_s507,)


@app.cell(hide_code=True)
def __(file_browser_s507, pd, timeserie_df):
    _s507_df_list = []
    for _file in file_browser_s507.value:
        _s507_df_list.append(pd.read_csv(_file.path, sep=';', skiprows=1, header=None, dtype=str))

    _corresp = {0: 'pdl',
                2: 'source',
                3: 'sens',
                4: 'code_postal',
                5: 'profil',
                6: 'start_date',
                7: 'end_date',
                13: 'Puissance'
               }
    # Concaténer tous les dataframes de timeserie
    if _s507_df_list:
        s507_df = pd.concat(_s507_df_list, ignore_index=True).rename(columns=_corresp)
        s507_df['start_date'] = pd.to_datetime(s507_df['start_date'], format="%d/%m/%Y")
        s507_df['end_date'] = pd.to_datetime(s507_df['end_date'], format="%d/%m/%Y")
        
        s507_df = s507_df.sort_values(['pdl', 'start_date'])
        #s507_df[['profil', 'sens', 'sous_profil']] = s507_df['code_profil'].str.split('_', expand=True)
    else:
        s507_df = pd.DataFrame()

    if not timeserie_df.empty:
        ...
    s507_df
    return (s507_df,)


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Interpretation

        Première chose, mettre des profils en face des PDL :
        """
    )
    return


@app.cell
def __(pd, s507_df):
    unique_profiles = s507_df.groupby('pdl')['profil'].unique()
    multiple_profiles = pd.DataFrame(unique_profiles[unique_profiles.apply(len) > 1])
    multiple_profiles
    return multiple_profiles, unique_profiles


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        PB : Chaque pdl à un profil qui varie au cours du temps. Il faut donc créer, de manière analogue à la présence du pdl dans notre périmètre, une liste des profils par pdl avec une période associée.

        Cela donne la table de correspondance *(pdl, temps, profil)* suivante :
        """
    )
    return


@app.cell(hide_code=True)
def __(pd, s507_df):
    def merge_profiles_in_group(df_group):
        # Assumptions :
        # df_group contient 'pdl', 'profil', 'start_date', 'end_date'
        
        # Conversion en datetime si nécessaire
        df_group['start_date'] = pd.to_datetime(df_group['start_date'])
        df_group['end_date'] = pd.to_datetime(df_group['end_date'])
        
        # Tri par start_date
        df_group = df_group.sort_values('start_date').reset_index(drop=True)
        
        merged_rows = []
        
        # Initialisation avec la première ligne du groupe
        current_profil = df_group.loc[0, 'profil']
        # current_start = df_group.loc[0, 'start_date']
        current_start = pd.Timestamp('2024-04-01')

        # Parcours des lignes du groupe (à partir de la 2ème)
        for i in range(1, len(df_group)):
            row = df_group.loc[i]
            # Si changement de profil, on enregistre la période précédente
            if row['profil'] != current_profil:
                # Fin de la période = (début de la nouvelle - 1 jour)
                # Ici, on impose la fin de la période précédente à la veille du début de cette nouvelle ligne
                prev_end = row['start_date'] - pd.Timedelta(days=1)
                
                merged_rows.append({
                    'pdl': df_group.loc[0, 'pdl'],
                    'profil': current_profil,
                    'start_date': current_start,
                    'end_date': prev_end
                })
                
                # Mise à jour du profil et du start de la nouvelle période
                current_profil = row['profil']
                current_start = row['start_date']
        
        # Une fois toutes les lignes parcourues, on ferme la dernière période
        # On prend la end_date de la dernière ligne du groupe
        last_end = df_group.loc[len(df_group)-1, 'end_date']
        merged_rows.append({
            'pdl': df_group.loc[0, 'pdl'],
            'profil': current_profil,
            'start_date': current_start,
            'end_date': last_end
        })
        
        return pd.DataFrame(merged_rows)
    profils_temporalises = s507_df.groupby('pdl', group_keys=False).apply(merge_profiles_in_group).reset_index(drop=True)
    profils_temporalises
    return merge_profiles_in_group, profils_temporalises


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Ajout de la colonne *profil* a la timeserie

        Pour chauqe ligne de la timeserie, on regarde dans la table de corespondance *(pdl, temps, profil)*
        quel est le profil du pdl.
        """
    )
    return


@app.cell
def __(profils_temporalises, timeserie_df):
    # Ajout profil
    def set_profil(row, _profils_temporalises):
        _pdl_periods = _profils_temporalises[_profils_temporalises['pdl'] == row['pdl']]
        for _, _period in _pdl_periods.iterrows():
            if (_period['start_date'] <= row['Date']) & (row['Date'] <= _period['end_date']):
                return _period['profil']

    timeserie_profilee = timeserie_df.copy()
    timeserie_profilee['profil'] = timeserie_profilee.apply(set_profil, _profils_temporalises=profils_temporalises, axis=1)

    # Extraire l'année et le mois
    timeserie_profilee['year_month'] = timeserie_profilee['Date'].dt.to_period('M')
    timeserie_profilee
    return set_profil, timeserie_profilee


@app.cell
def __(mo):
    mo.md(
        """
        # Compilation des résultats

        On a maintenant, pour chacune des lignes de la timesérie, ajouté une Marque et un profil (cf chapitres précédents). 

        Chaque ligne de notre timesérie représente donc maintenant une quantité d'énergie consommée par tel PDL sur un pas donné, avec un profil associé et une Marque identifiée.
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Consommations par profils

        En regroupant par profils et par mois, puis en effectuant la somme des valeurs regroupées, on obtient ce tableau des consommations mensuelles par profil :
        """
    )
    return


@app.cell
def __(timeserie_profilee):
    # Grouper par profil et par year_month, et sommer les Valeur
    conso_mensuelles = (timeserie_profilee[timeserie_profilee['Actif']].groupby(['profil', 'year_month'])['Valeur'].sum().unstack('profil') / 1e6).round(3)

    conso_mensuelles['total'] = conso_mensuelles.sum(axis=1)
    conso_mensuelles
    return (conso_mensuelles,)


if __name__ == "__main__":
    app.run()
