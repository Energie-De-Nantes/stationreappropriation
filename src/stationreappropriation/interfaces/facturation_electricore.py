import marimo

__generated_with = "0.11.12"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np

    from datetime import date
    from pathlib import Path

    from electriflux.simple_reader import process_flux

    from stationreappropriation import load_prefixed_dotenv
    from stationreappropriation.utils import gen_previous_month_boundaries, gen_last_months

    env = load_prefixed_dotenv(prefix='SR_')
    flux_path = Path('~/data/flux_enedis_v2/').expanduser()
    flux_path.mkdir(parents=True, exist_ok=True)

    radio = mo.ui.radio(options=gen_last_months(), label='Choisi le Mois a traiter')
    radio
    return (
        Path,
        date,
        env,
        flux_path,
        gen_last_months,
        gen_previous_month_boundaries,
        load_prefixed_dotenv,
        mo,
        np,
        pd,
        process_flux,
        radio,
    )


@app.cell(hide_code=True)
def _(gen_previous_month_boundaries, mo, radio):
    default_start, default_end = radio.value if radio.value is not None else gen_previous_month_boundaries()
    start_date_picker = mo.ui.date(value=default_start)
    end_date_picker = mo.ui.date(value=default_end)
    mo.md(
        f"""
        Tu peux aussi choisir directement les dates de début {start_date_picker} et de fin {end_date_picker}\n
        """
    )
    return default_end, default_start, end_date_picker, start_date_picker


@app.cell(hide_code=True)
def _(end_date_picker, pd, start_date_picker):
    from zoneinfo import ZoneInfo
    PARIS_TZ = ZoneInfo("Europe/Paris")
    deb = pd.to_datetime(start_date_picker.value).tz_localize(PARIS_TZ)
    fin = pd.to_datetime(end_date_picker.value).tz_localize(PARIS_TZ)
    return PARIS_TZ, ZoneInfo, deb, fin


@app.cell
def _(mo):
    mo.md(r"""# Chargement des flux""")
    return


@app.cell(hide_code=True)
def _(env, flux_path, mo):
    from stationreappropriation.marimo_utils import download_with_marimo_progress as _dl
    _processed, _errors = _dl(env, ['R15', 'R151', 'C15', 'F15', 'F12'], flux_path)
    mo.md(f"Processed #{len(_processed)} files, with #{len(_errors)} erreurs")
    return


@app.cell(hide_code=True)
def _(flux_path, process_flux):
    from electricore.inputs.flux import lire_flux_c15

    historique = lire_flux_c15(process_flux('C15', flux_path / 'C15'))
    return historique, lire_flux_c15


@app.cell(hide_code=True)
def _(flux_path, process_flux):
    from electricore.inputs.flux import lire_flux_r151

    relevés = lire_flux_r151(process_flux('R151', flux_path / 'R151'))
    return lire_flux_r151, relevés


@app.cell
def _(historique, mo, relevés):
    mo.accordion({"relevés": relevés, 'historique': historique}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Calculs énergies et Taxes""")
    return


@app.cell
def _(deb, fin, historique, relevés):
    from electricore.core.services import facturation
    factu = facturation(deb, fin, historique, relevés)
    factu
    return factu, facturation


@app.cell(hide_code=True)
def _():
    # Odoo
    return


@app.cell(hide_code=True)
def _():
    ## Récupération des abonnements à facturer
    return


@app.cell
def _(env, mo):
    from stationreappropriation.odoo import get_enhanced_draft_orders
    draft_orders = get_enhanced_draft_orders(env)
    _stop_msg = mo.callout(mo.md(
        f"""
        ## ⚠ Aucun abonnement à facturer trouvé sur [{env['ODOO_URL']}]({env['ODOO_URL']}web#action=437&model=sale.order&view_type=kanban). ⚠ 
        Ici ne sont prises en comptes que les cartes dans la colonne **Facture brouillon créée**, et le programme n'en trouve pas.
        Le processus de facturation ne peut pas continuer en l'état.

        Plusieurs causes possibles :

        1. Le processus de facturation n'a pas été lancé dans Odoo. Go le lancer. 
        2. Toutes les cartes abonnement ont déjà été déplacées dans une autre colonne. Si tu souhaite néanmoins re-mettre à jour un des abonnements, il suffit de redéplacer sa carte dans la colonne Facture brouillon créée. Attention, ça va écraser les valeurs de sa facture.
        """), kind='warn')

    mo.stop(draft_orders.empty, _stop_msg)
    draft_orders
    return draft_orders, get_enhanced_draft_orders


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Fusion des données Enedis et Odoo""")
    return


@app.cell
def _(draft_orders, end_date_picker, start_date_picker, taxes):
    _required_cols = ['HP', 'HC', 'BASE', 'j', 'd_date', 'f_date', 'Type_Compteur', 'Num_Compteur', 'Num_Depannage', 'pdl', 'turpe_fix', 'turpe_var', 'turpe', 'missing_data']
    merged_data = draft_orders.merge(taxes[_required_cols], left_on='x_pdl', right_on='pdl', how='left')
    days_in_month = (end_date_picker.value - start_date_picker.value).days
    merged_data['update_dates'] = merged_data['j'] != days_in_month
    merged_data['missing_data'] = merged_data['missing_data'].astype(bool).fillna(True)
    merged_data['something_wrong'] = (merged_data['missing_data'] == True) & (merged_data['x_lisse'] == False)

    merged_data
    return days_in_month, merged_data


if __name__ == "__main__":
    app.run()
