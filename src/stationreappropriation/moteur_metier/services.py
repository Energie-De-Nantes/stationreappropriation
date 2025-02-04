import pandas as pd
from .consos import qui_quoi_quand, ajout_R151, ajout_par_defaut, calcul_energie
from .turpe import get_applicable_rules, compute_turpe
from .formattage import supprimer_colonnes, fusion_des_sous_periode, validation

def energies_et_taxes(deb: pd.Timestamp, fin: pd.Timestamp, c15: pd.DataFrame, r151: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule les énergies et les taxes pour une période donnée, sur l'ensemble du périmètre des flux c15 et r151
    """
    
    alors = qui_quoi_quand(deb, fin, c15)
    indexes = ajout_R151(deb, fin, alors, r151)
    complet = ajout_par_defaut(deb, fin, indexes)
    energies = calcul_energie(complet)
    energies['Puissance_Souscrite'] = pd.to_numeric(energies['Puissance_Souscrite'])
    rules = get_applicable_rules(deb, fin)
    turpe = compute_turpe(entries=energies, rules=rules)
    final = validation(
        supprimer_colonnes(
        fusion_des_sous_periode(turpe)))
    return final.round(2)