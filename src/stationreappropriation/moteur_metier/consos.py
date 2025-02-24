import pandas as pd
from pathlib import Path
import warnings
from zoneinfo import ZoneInfo
from pandas import DataFrame

# Quoiquonfé ? on a deux timestamps start et end qui délimitent la période de calcul, on veut déterminer plusieurs trucs :
# 1) les couples (usager.e, pdl) dont on doit estimer la consommation
# 2) les éventuels entrées/sorties dans la période de calcul
# 3) les éventuels MCT (code pour dire changement de puissance ou de Formule Tarifaire d'acheminement aka FTA) une MCT n'impacte pas le calcul des consos mais celui des taxes ultérieurement
#       Si MCT, on veut créer pour le couple (usager.e, pdl) correspondant deux lignes de consommation, une avant la MCT et une après

def qui_quoi_quand(deb: pd.Timestamp, fin: pd.Timestamp, c15: DataFrame) -> DataFrame:
    """
    On veut les couples (usager.e, pdl) dont on doit estimer la consommation.

    Note : on s'intéresse aux couples (usager.e, pdl) qui sont dans le C15, dont le statut actuel est 'EN SERVICE' et celleux dont le statut est 'RESILIE' et dont la date de résiliation est dans la période de calcul.
    Args:
        deb (pd.Timestamp): début de la période de calcul
        fin (pd.Timestamp): fin de la période de calcul
        c15 (DataFrame): Flux de facturation C15 sous forme de DataFrame
    """
    # On ne veut pas considérer les lignes de C15 qui sont advenues après la fin de la période de calcul
    c15_fin = c15[c15['Date_Evenement'] <= fin]

    c15_periode = c15_fin[c15_fin['Date_Evenement'] >= deb]
    situation_fin = c15_fin.copy().sort_values(by='Date_Evenement', ascending=False).drop_duplicates(subset=['pdl', 'Ref_Situation_Contractuelle'], keep='first')

    # 1)
    # on s'intéresse aux couples (usager.e, pdl) qui sont dans le C15, dont le statut actuel est 'EN SERVICE' et celleux dont le statut est 'RESILIE' et dont la date de résiliation est dans la période de calcul.
    _masque = (situation_fin['Etat_Contractuel'] == 'EN SERVICE') | ((situation_fin[ 'Etat_Contractuel'] == 'RESILIE') & (situation_fin['Date_Evenement'] >= deb))

    colonnes_releve = ['Date_Releve', 'Nature_Index', 'HP', 'HC', 'HCH', 'HPH', 'HPB', 'HCB', 'BASE']
    colonnes_evenement = ['Date_Derniere_Modification_FTA', 'Evenement_Declencheur', 'Type_Evenement', 'Date_Evenement', 'Ref_Demandeur', 'Id_Affaire']
    base_de_travail = (
        situation_fin[_masque]
        .drop(columns=colonnes_releve+colonnes_evenement)
        .copy()
        # .set_index('Ref_Situation_Contractuelle')
    )

    # 2) Entrées/sorties
    entrees = (
        c15_periode[c15_periode['Evenement_Declencheur']
        .isin(['MES', 'PMES', 'CFNE'])]
        .set_index('Ref_Situation_Contractuelle')[colonnes_releve]
        .add_suffix('_deb')
    )
    entrees['source_releve_deb'] = 'entree_C15'
    sorties = (
        c15_periode[c15_periode['Evenement_Declencheur']
        .isin(['RES', 'CFNS'])]
        .set_index('Ref_Situation_Contractuelle')[colonnes_releve]
        .add_suffix('_fin')
    )
    sorties['source_releve_fin'] = 'sortie_C15'
    # Fusion avec la base de travail
    base_de_travail = (
        base_de_travail
        .merge(entrees, how='left', left_on='Ref_Situation_Contractuelle', right_index=True)
        .merge(sorties, how='left', left_on='Ref_Situation_Contractuelle', right_index=True)
    )

    # 3) Prise en compte des MCT.
    mct = (
        c15_periode[c15_periode['Evenement_Declencheur']
        .isin(['MCT'])]
        # .set_index('Ref_Situation_Contractuelle')[colonnes_releve]
    )

    base_de_travail = diviser_lignes_mct(base_de_travail, mct, colonnes_releve)
    
    # base_de_travail = ajout_par_defaut(deb, fin, 'R151', base_de_travail)

    # base_de_travail = calcul_nb_jours(base_de_travail)

    return base_de_travail

def diviser_lignes_mct(base_df: pd.DataFrame, mct_df: pd.DataFrame, colonnes_releve: list) -> pd.DataFrame:
    """
    Divise les lignes ayant une MCT en deux périodes : avant et après MCT.
    
    Args:
        base_df (pd.DataFrame): DataFrame original contenant les lignes à diviser
        mct_df (pd.DataFrame): DataFrame contenant les données MCT avec Ref_Situation_Contractuelle
        colonnes_releve (list): Liste des colonnes de relevé à considérer
    
    Returns:
        pd.DataFrame: DataFrame avec les lignes MCT divisées en deux périodes
    """
    # Identification des lignes avec MCT
    lignes_avec_mct = (
        base_df.index[
            base_df['Ref_Situation_Contractuelle']
            .isin(mct_df['Ref_Situation_Contractuelle'])
        ]
    )        
    if len(lignes_avec_mct) == 0:
        return base_df

    # Création des deux nouveaux jeux de lignes
    lignes_avant_mct = base_df.loc[lignes_avec_mct].copy()
    lignes_apres_mct = base_df.loc[lignes_avec_mct].copy()

    # Préparation des colonnes MCT avec les bons suffixes
    mct_fin = mct_df[['Ref_Situation_Contractuelle'] + colonnes_releve].copy()
    mct_fin.columns = ['Ref_Situation_Contractuelle'] + [f'{col}_fin' for col in colonnes_releve]
    
    mct_deb = mct_df[['Ref_Situation_Contractuelle'] + colonnes_releve].copy()
    mct_deb.columns = ['Ref_Situation_Contractuelle'] + [f'{col}_deb' for col in colonnes_releve]

    # Suppression des anciennes colonnes de fin/début avant la fusion
    colonnes_a_supprimer_fin = [f'{col}_fin' for col in colonnes_releve]
    colonnes_a_supprimer_deb = [f'{col}_deb' for col in colonnes_releve]
    
    lignes_avant_mct = lignes_avant_mct.drop(columns=colonnes_a_supprimer_fin, errors='ignore')
    lignes_apres_mct = lignes_apres_mct.drop(columns=colonnes_a_supprimer_deb, errors='ignore')

    # Fusion avec les données MCT
    lignes_avant_mct = lignes_avant_mct.merge(
        mct_fin,
        on='Ref_Situation_Contractuelle'
    )
    lignes_avant_mct['source_releve_fin'] = 'MCT'

    lignes_apres_mct = lignes_apres_mct.merge(
        mct_deb,
        on='Ref_Situation_Contractuelle'
    )
    lignes_apres_mct['source_releve_deb'] = 'MCT'

    # Construction du DataFrame final
    return pd.concat([
        base_df.drop(index=lignes_avec_mct),
        lignes_avant_mct,
        lignes_apres_mct
    ])

def ajout_par_defaut(deb: pd.Timestamp, fin: pd.Timestamp, df: DataFrame) -> DataFrame:
    """
    Ajoute les valeurs par défaut pour les colonnes 'Date_Releve_deb', 'Date_Releve_fin', 'source_releve_deb', et 'source_releve_fin'.
    Args:
        df (DataFrame): Le DataFrame à traiter.
        deb (pd.Timestamp): La date de début par défaut.
        fin (pd.Timestamp): La date de fin par défaut.
    Returns:
        DataFrame: Le DataFrame avec les valeurs par défaut ajoutées.
    """
    df['Date_Releve_deb'] = df['Date_Releve_deb'].fillna(deb)
    df['Date_Releve_fin'] = df['Date_Releve_fin'].fillna(fin)
    return df

def calcul_nb_jours(df: DataFrame) -> DataFrame:
    """
    Calcule le nombre de jours entre 'Date_Releve_deb' et 'Date_Releve_fin' et ajoute le résultat dans la colonne 'nb_jours'.

    Note: 
        On veut le nb de jours entre les deux dates, avec le premier jour inclus et le dernier jour exclu.
        Si les deux dates sont égales, le nombre de jours est de 0.

    Args:
        df (DataFrame): Le DataFrame à traiter.
    Returns:
        DataFrame: Le DataFrame avec la colonne 'nb_jours' ajoutée.
    """
    df['nb_jours'] = (df['Date_Releve_fin'] - df['Date_Releve_deb']).dt.days
    return df

def ajout_R151(deb: pd.Timestamp, fin: pd.Timestamp, df: DataFrame, r151: DataFrame) -> DataFrame:
    """
    Vient ajouter les relevés issus du R151 la ou il n'y a pas de relevé.

    Note : 
       Il peut rester des trous dans les relevés, même après cette opération.

    Args:
        deb (pd.Timestamp): La date de début de la période.
        fin (pd.Timestamp): La date de fin de la période.
        df (DataFrame): Le DataFrame à traiter.
    Returns:
        DataFrame: Le DataFrame enrichie avec les relevés issus du R151 ajoutés.
    """

    colonnes_releve_r151 = ['Date_Releve', 'HP', 'HC', 'HCH', 'HPH', 'HPB', 'HCB', 'BASE']
    r151['Date_Releve'] = (
        pd.to_datetime(r151['Date_Releve'])
        .dt.tz_localize(None)
        .dt.tz_localize(ZoneInfo("Europe/Paris"))
    )
    # Pour les débuts de période
    
    releves_deb = (
        r151[r151['Date_Releve'].dt.date == deb.date()]
        .set_index('pdl')[colonnes_releve_r151]
        .assign(source_releve='R151')
        .add_suffix('_deb')
    )
    
    releves_fin = (
        r151[r151['Date_Releve'].dt.date == fin.date()]
        .set_index('pdl')[colonnes_releve_r151]
        .assign(source_releve='R151')
        .add_suffix('_fin')
    )
    r = df.copy()
    # Trouver les valeurs à mettre à jour dans df
    masque_deb = (r['source_releve_deb'].isna()) | (r['source_releve_deb'] == 'R151')
    masque_fin = (r['source_releve_fin'].isna()) | (r['source_releve_fin'] == 'R151')

    # MAJ des valeurs 
    r.loc[masque_deb, releves_deb.columns] = (
        releves_deb
        .reindex(r.loc[masque_deb, 'pdl'])
        .values
    )
    r.loc[masque_fin, releves_fin.columns] = (
        releves_fin
        .reindex(r.loc[masque_fin, 'pdl'])
        .values
    )
    return r

def calcul_energie(df: DataFrame) -> DataFrame:
    """
    Calcul de la consommation d'énergie en kWh, par cadran
    
    """
    # Calcul de la consommation d'énergie
    r = df.copy()

    cadrans = ['HPH', 'HPB', 'HCH', 'HCB', 'HP', 'HC',  'BASE']
    df['releve_manquant'] = df[['source_releve_fin', 'source_releve_deb']].isna().any(axis=1)
    for c in cadrans:
        colonne_deb = f"{c}_deb"
        colonne_fin = f"{c}_fin"
        if colonne_deb in r.columns and colonne_fin in r.columns:
            # Vérifie que les deux sources de relevé sont non nulles
            masque_valide = ~df['releve_manquant']
            diff = r.loc[masque_valide, colonne_fin] - r.loc[masque_valide, colonne_deb]
            
            pdls_negatifs = r.loc[masque_valide][diff < 0]['pdl'].tolist()
            if pdls_negatifs:
                warnings.warn(f"Valeurs négatives détectées pour le compteur {c} "
                            f"sur les PDLs: {pdls_negatifs}")
            
            # Initialise la colonne avec NaN
            r[c] = pd.NA
            # Applique le calcul uniquement où mask_valide est True
            r.loc[masque_valide, c] = diff
            
    # Calcul du nombre de jours entre les deux relevés
    r['j'] = (r['Date_Releve_fin'] - r['Date_Releve_deb']).dt.days

    # Calculer HP et HC en prenant la somme des colonnes correspondantes
    r['HP'] = r[['HPH', 'HPB', 'HP']].sum(axis=1, min_count=1)
    r['HC'] = r[['HCH', 'HCB', 'HC']].sum(axis=1, min_count=1)

    # Calculer BASE uniquement là où BASE est NaN
    r.loc[r['BASE'].isna(), 'BASE'] = r[['HP', 'HC']].sum(axis=1, min_count=1)
    
    return r
    