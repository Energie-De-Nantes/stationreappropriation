import marimo

__generated_with = "0.11.17"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def choix_mois_facturation():
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


@app.cell
def _():
    # gen_last_months()
    return


@app.cell(hide_code=True)
def choix_dates_facturation(gen_previous_month_boundaries, mo, radio):
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
def conversion_dates(end_date_picker, pd, start_date_picker):
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
def _(mo):
    mo.md(r"""## Téléchargement et extraction des zip""")
    return


@app.cell(hide_code=True)
def telechargement_flux(env, flux_path, mo):
    from stationreappropriation.marimo_utils import download_with_marimo_progress as _dl
    _processed, _errors = _dl(env, ['R15', 'R151', 'C15', 'F15', 'F12'], flux_path)
    mo.md(f"Processed #{len(_processed)} files, with #{len(_errors)} erreurs")
    return


@app.cell(hide_code=True)
def _():
    ## Interprétation des flux
    return


@app.cell(hide_code=True)
def chargement_perimetre(flux_path, process_flux):
    from electricore.inputs.flux import lire_flux_c15

    historique = lire_flux_c15(process_flux('C15', flux_path / 'C15'))
    return historique, lire_flux_c15


@app.cell(hide_code=True)
def chargement_releves(flux_path, process_flux):
    from electricore.inputs.flux import lire_flux_r151

    relevés = lire_flux_r151(process_flux('R151', flux_path / 'R151'))
    return lire_flux_r151, relevés


@app.cell(hide_code=True)
def visualisation_donnees_metier(historique, mo, relevés):
    mo.accordion({"relevés": relevés, 'historique': historique}, lazy=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Calculs énergies et Taxes""")
    return


@app.cell(hide_code=True)
def identification_problemes_metier(deb, fin, historique):
    from electricore.core.périmètre.fonctions import extraire_historique_à_date, extraire_modifications_impactantes

    mci = extraire_modifications_impactantes(deb=deb, historique=extraire_historique_à_date(fin=fin, historique=historique))
    mci
    return extraire_historique_à_date, extraire_modifications_impactantes, mci


@app.cell(hide_code=True)
def calcul_energies_taxes(deb, fin, historique, relevés):
    from electricore.core.services import facturation
    factu = facturation(deb, fin, historique, relevés)
    factu
    return factu, facturation


@app.cell(hide_code=True)
def _(mo):
    mo.md("""# Odoo""")
    return


@app.cell(hide_code=True)
def _():
    ## Récupération des abonnements à facturer
    return


@app.cell(hide_code=True)
def chargement_donnees_odoo(env, mo):
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
    mo.md(r"""# Fusion des données Enedis et Odoo""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Préparation des données métier

        Cette adaptation est nécessaire pour connecter la nouvelle version de electricore à la partie Odoo de stationréappropriation. Il y aura une correspondance directe plus tard.
        """
    )
    return


@app.cell(hide_code=True)
def preparation_metier(factu):
    required_cols = ['HP', 'HC', 'BASE', 'j', 'd_date', 'f_date', 'Type_Compteur', 'Num_Compteur', 'Num_Depannage', 'pdl', 'turpe_fix', 'turpe_var', 'turpe', 'missing_data']

    # On renomme quelques trucs pour se faciliter la vie pour l'instant : 

    métier = factu.copy()
    métier = métier.rename(columns={
        'Date_Releve_deb': 'd_date',
        'Date_Releve_fin': 'f_date',
        'turpe_fixe': 'turpe_fix'})
    métier['missing_data'] = ~métier['Energie_Calculee']
    import pandera as pa
    from typing import List, Optional
    class ModèleMétier(pa.DataFrameModel):
        HP: pa.typing.Series[float] = pa.Field(nullable=True)
        HC: pa.typing.Series[float] = pa.Field(nullable=True)
        BASE: pa.typing.Series[float] = pa.Field(nullable=True)
        j: pa.typing.Series[int] = pa.Field(nullable=True)
        d_date: pa.typing.Series = pa.Field() # Jsais plus ce qui est attendu
        f_date: pa.typing.Series = pa.Field() # Jsais plus ce qui est attendu
        Type_Compteur: pa.typing.Series[str] = pa.Field()
        Num_Compteur: pa.typing.Series[str] = pa.Field()
        Num_Depannage: pa.typing.Series[str] = pa.Field()
        pdl: pa.typing.Series[str] = pa.Field()
        turpe_fix: pa.typing.Series[float] = pa.Field(nullable=True)
        turpe_var: pa.typing.Series[float] = pa.Field(nullable=True)
        turpe: pa.typing.Series[float] = pa.Field(nullable=True)
        missing_data: pa.typing.Series[bool] = pa.Field()

    métier = ModèleMétier.validate(métier)
    return List, ModèleMétier, Optional, métier, pa, required_cols


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Fusion données métier et données facturation""")
    return


@app.cell
def fusion_metier_odoo(
    draft_orders,
    end_date_picker,
    métier,
    required_cols,
    start_date_picker,
):
    merged_data = draft_orders.merge(métier[required_cols], left_on='x_pdl', right_on='pdl', how='left')
    days_in_month = (end_date_picker.value - start_date_picker.value).days
    merged_data['update_dates'] = merged_data['j'] != days_in_month
    merged_data['missing_data'] = merged_data['missing_data'].astype(bool).fillna(True)
    merged_data['something_wrong'] = (merged_data['missing_data'] == True) & (merged_data['x_lisse'] == False)

    merged_data
    return days_in_month, merged_data


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Préparation des données avant envoi""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Abonnements""")
    return


@app.cell(hide_code=True)
def _(mo):
    slider = mo.ui.slider(start=1, stop=10, step=1, value=5, label='Pourcentage de couverture des tests')
    slider
    return (slider,)


@app.cell(hide_code=True)
def preparation_abonnements(merged_data, mo, np, pd, slider):
    _orders = pd.DataFrame(merged_data['sale.order_id'].copy())
    _orders['x_invoicing_state'] = np.where(np.random.rand(len(_orders)) < slider.value/100, 'populated', 'checked')
    _orders.loc[merged_data['something_wrong'] == True, 'x_invoicing_state'] = 'draft'
    _orders.rename(columns={'sale.order_id':'id'}, inplace=True)
    orders = _orders.to_dict(orient='records')
    mo.accordion({
        'Abonnements':_orders, 
    })
    return (orders,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""### Factures""")
    return


@app.cell(hide_code=True)
def preparation_factures(merged_data, mo):
    _invoices = merged_data[['last_invoice_id', 
                            'turpe', 
                            'd_date', 'f_date', 
                            'Type_Compteur', 
                            'Num_Compteur']].copy()

    _invoices['d_date'] = _invoices['d_date'].dt.strftime('%Y-%m-%d')
    _invoices['f_date'] = _invoices['f_date'].dt.strftime('%Y-%m-%d')

    _to_rename = {
        'last_invoice_id':'id',
        'turpe':'x_turpe',
        'd_date':'x_start_invoice_period',
        'f_date':'x_end_invoice_period',
        'Type_Compteur':'x_type_compteur',
        'Num_Compteur':'x_num_serie_compteur',
    }

    _invoices.rename(columns=_to_rename, inplace=True)
    invoices = _invoices.to_dict(orient='records')

    mo.accordion({
        'Factures':_invoices, 
    })
    return (invoices,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Lignes de facture""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Énergies Consommées

        On ne met à jour que les non-lissés, dont on dispose des données.
        """
    )
    return


@app.cell(hide_code=True)
def preparation_lignes_energies(merged_data, pd):
    # On s'intéresse uniquement les données d'énergies qu'il faut mettre à jour
    update_conso_df = merged_data[(~merged_data['x_lisse']) & (merged_data['something_wrong']==False)].copy()

    hc = pd.DataFrame()
    hp = pd.DataFrame()
    base = pd.DataFrame()
    if 'line_id_HC' in update_conso_df.columns:
        hc = update_conso_df[update_conso_df['line_id_HC'].notna()][['line_id_HC', 'HC']]
        hc = hc.dropna(subset=['HC'])
        hc.rename(columns={'line_id_HC': 'id', 'HC': 'quantity'}, inplace=True)

    if 'line_id_HP' in update_conso_df.columns:
        hp = update_conso_df[update_conso_df['line_id_HP'].notna()][['line_id_HP', 'HP']]
        hp = hp.dropna(subset=['HP'])
        hp.rename(columns={'line_id_HP': 'id', 'HP': 'quantity'}, inplace=True)

    if 'line_id_Base' in update_conso_df.columns:
        base = update_conso_df[update_conso_df['line_id_Base'].notna()][['line_id_Base', 'BASE']]
        base = base.dropna(subset=['BASE'])
        base.rename(columns={'line_id_Base': 'id', 'BASE': 'quantity'}, inplace=True)
    return base, hc, hp, update_conso_df


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Jours d'abonnement

        On ne met à jour que si pas lissé ou lissé entré ou sorti dans la période
        """
    )
    return


@app.cell(hide_code=True)
def preparation_lignes_abonnements(merged_data):
    do_update_qty = (~(merged_data['x_lisse'] == True) | (merged_data['update_dates'] == True))
    abo = merged_data[do_update_qty][['line_id_Abonnements', 'j']]
    abo = abo.dropna(subset=['line_id_Abonnements', 'j'])
    abo.rename(columns={'line_id_Abonnements': 'id', 'j': 'quantity'}, inplace=True)
    return abo, do_update_qty


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Convertion DataFrame vers une liste de lignes

        On va ajouter toutes les lignes, hp, hc, base et abonnement dans une même liste, qui sera ensuite envoyée d'un bloc à Odoo.
        """
    )
    return


@app.cell(hide_code=True)
def serialisation_lignes_factures(abo, base, hc, hp):
    lines = []
    lines += base.to_dict(orient='records')
    lines += hp.to_dict(orient='records')
    lines += hc.to_dict(orient='records')
    lines += abo.to_dict(orient='records')
    return (lines,)


@app.cell
def _(mo):
    mo.md(r"""# Visualisation des données à envoyer""")
    return


@app.cell
def visualisation_lignes_facture(abo, base, hc, hp, lines, mo):
    mo.accordion({
        'HC':hc, 
        'HP':hp, 
        'base':base, 
        'jours':abo,
        'Lignes de factures': lines
    })
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Envoi des données à Odoo""")
    return


@app.cell
def declanchement_envoi_vers_odoo(mo):
    red_button = mo.ui.run_button(kind='danger', label=f'Écrire dans la base Odoo')
    red_button
    return (red_button,)


@app.cell
def envoi_vers_odoo(env, invoices, lines, mo, orders, red_button):
    from stationreappropriation.odoo import OdooConnector

    mo.stop(not red_button.value)

    with OdooConnector(config=env, sim=False) as _odoo:
        _odoo.update("sale.order", orders)
        _odoo.update("account.move", invoices)
        _odoo.update("account.move.line", lines)
    return (OdooConnector,)


if __name__ == "__main__":
    app.run()
