import pandas as pd
from .consos import qui_quoi_quand, ajout_R151, calcul_energie
from .turpe import get_applicable_rules, compute_turpe


def energies_et_taxes(deb: pd.Timestamp, fin: pd.Timestamp, c15: pd.DataFrame, r151: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule les énergies et les taxes pour une période donnée, sur l'ensemble du périmètre des flux c15 et r151
    """
    
    alors = qui_quoi_quand(deb, fin, c15)
    indexes = ajout_R151(deb, fin, alors, r151)
    energies = calcul_energie(indexes)
    energies['Puissance_Souscrite'] = pd.to_numeric(energies['Puissance_Souscrite'])
    rules = get_applicable_rules(deb, fin)
    turpe = compute_turpe(entries=energies, rules=rules).round(2)

    return turpe