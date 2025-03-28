from stationreappropriation.odoo import OdooConnector

import pandas as pd
from pandas import DataFrame, Series

def get_valid_subscriptions_pdl(config: dict) -> DataFrame:
    # Initialiser OdooAPI avec le dict de configuration
    with OdooConnector(config=config, sim=True) as odoo:
        # Lire les abonnements Odoo valides en utilisant la fonction search_read
        valid_subscriptions = odoo.search_read('sale.order', 
            filters=[[['is_subscription', '=', True], 
                        ['is_expired', '=', False], 
                        ['state', '=', 'sale'], 
                        ['subscription_state', '=', '3_progress']]], 
            fields=['name', 'x_pdl', 'x_lisse'])
    
        return valid_subscriptions.rename(columns={'x_pdl': 'pdl', 'x_lisse': 'lisse'})

def get_pdls(config: dict) -> DataFrame:
    # Initialiser OdooAPI avec le dict de configuration
    with OdooConnector(config=config, sim=True) as odoo:
    
        # Lire les abonnements Odoo valides et passés en utilisant la fonction search_read
        all_subscriptions = odoo.search_read('sale.order', 
            filters=[[['is_subscription', '=', True],
                      ['state', '=', 'sale'], ]],
            fields=['x_pdl'])
    
        return all_subscriptions.rename(columns={'x_pdl': 'pdl',})
    
def get_enhanced_draft_orders(config: dict) -> DataFrame:
    with OdooConnector(config=config, sim=True) as odoo:
        draft_orders = odoo.search_read('sale.order', 
                                        filters=[[['state', '=', 'sale'], 
                                                  ['x_invoicing_state', '=', 'draft'],
                                                  ['subscription_state', '=', '3_progress']]],
                                        fields=['id',
                                                'display_name', 
                                                'x_pdl', 
                                                'invoice_ids', 
                                                'x_lisse',])

        if draft_orders.empty:
            return draft_orders
        # Ici, on filtre la liste des facture en prenant uniquement celle a l'identifiant le plus élevé
        # Ptet plus tard on voudra prendre en compte d'autres critères
        draft_orders['last_invoice_id'] = draft_orders['invoice_ids'].apply(lambda x: max(x) if x else None)

        draft_invoices = odoo.read('account.move', ids=draft_orders['last_invoice_id'].to_list(), fields=['invoice_line_ids', 'state'])
        if draft_invoices.empty:
            return draft_orders
        
        draft_invoices = draft_invoices[draft_invoices['state'] == 'draft']
        
        merged = draft_orders.merge(draft_invoices[['account.move_id', 'invoice_line_ids']], left_on='last_invoice_id', right_on='account.move_id', how='left')
        
        compl = _add_cat_fields(config, merged, [])
        
        if compl.empty:
            return compl
        compl.drop(columns=['account.move_id', 'invoice_ids', 'invoice_line_ids'], inplace=True)
        return compl
    
def _add_cat_fields(config: dict, data: DataFrame, fields: list[str])-> DataFrame:
    """
    Add one category column to the data frame for each invoice line, set with the corresponding line id.

    Args:
        data (DataFrame): The input data frame.
        fields (List[str]): The list of category fields to add.

    Returns:
        DataFrame: The input data frame with the added category fields.

    Raises:
        ValueError: If the input data frame does not have the required columns.

    After checking that the input data frame has the required columns,
    Then fetchs the lines of each invoice with invoice_line_ids key.
    Then fetchs the product of each line with product_id key found for each line.
    Then explodes each invoice line into a separate row.
    Then adds the cat columns from the fetcheds products.
    We now have all the data, but we need to return to one row for each invoice.
    We can do this by pivoting the data, and then merging the pivoted data with the original data frame.
    """
    if 'invoice_line_ids' not in data.columns:
        raise ValueError(f'No invoice_line_ids found in {data.columns}')
    with OdooConnector(config=config, sim=True) as odoo:
        # On explose les lignes de factures en lignes sépsarées, aka on duplique chaque ligne autant de fois qu'il y a des valeurs dans la liste 'invoice_line_ids'  
        df_exploded = data.explode('invoice_line_ids')
        df_exploded.to_clipboard()
        print(len(df_exploded['invoice_line_ids']))
        # On récupère les id produits de chaque ligne de facture
        lines = odoo.read('account.move.line', 
                          ids=df_exploded['invoice_line_ids'].to_list(),
                          fields=['id', 'product_id'])
        if lines.empty:
            return DataFrame()
        lines = lines.dropna(subset=['product_id'])
        lines['product_id'] = _get_many2one_id_list(lines['product_id'])

        # On récupère les catégories de chaque produit unique
        prods = odoo.read('product.product',
                          ids=lines['product_id'].drop_duplicates().to_list(),
                          fields=['id', 'categ_id'])
        prods['cat'] = _get_many2one_text_serie(prods['categ_id']).apply(lambda x: x.split(' ')[-1]).astype(str)

        # On fusionne lines et prods
        lines = lines.merge(prods[['product.product_id', 'cat']], left_on='product_id', right_on='product.product_id', how='left')
        
        # On fusionne dans df_exploded, en supprimant les lignes de df_exploded qui ne correspondent pas a un produit. 
        df_exploded = df_exploded.merge(lines[['account.move.line_id', 'cat']], left_on='invoice_line_ids', right_on='account.move.line_id', how='left')
        df_exploded.drop(columns=['invoice_line_ids'], inplace=True)
        df_exploded.dropna(subset=['account.move_id'], inplace=True)
        df_exploded['account.move_id'] = df_exploded['account.move_id'].astype('Int64')
        df_exploded['account.move.line_id'] = df_exploded['account.move.line_id'].astype('Int64')

        is_pe = df_exploded['cat'] == 'Prestation-Enedis'

        # Pour les catégories autres que 'Prestation-Enedis', pivotons normalement aka ça va créer des colonnes dédiées pour chaque catégorie
        df_pivot = df_exploded[~is_pe].pivot_table(index='account.move_id', columns='cat', values='account.move.line_id').reset_index()
        df_pivot.columns = ['account.move_id'] + [f'line_id_{x}' for x in df_pivot.columns if x != 'account.move_id']

        # Convert float columns back to int
        for col in df_pivot.columns:
            if col.startswith('line_id_'):
                print(col)
                df_pivot[col] = df_pivot[col].astype('Int64')
        
        # Pour 'Prestation-Enedis', agrégeons les valeurs dans une liste (il peux y en avoir plusieurs, contrairement aux autres catégories)
        df_all = df_exploded[is_pe].groupby('account.move_id')['account.move.line_id'].apply(list).reset_index()
        df_all.columns = ['account.move_id', 'line_id_Prestation-Enedis']

        # Fusionnons d'abord les DataFrames pivotés normalement et 'Prestation-Enedis'
        df_merged = pd.merge(df_pivot, df_all, on='account.move_id', how='left')

        # Ensuite, on fusionne le résultat avec le DataFrame original
        df_final = pd.merge(data, df_merged, on='account.move_id', how='left')

        return df_final
    
def _get_many2one_id_list(serie: Series) -> list:
    """
    Les id récupérés d'odoo sont sous la forme [id, texte], on ne garde que l'id
    """
    return serie.apply(lambda x: x[0] if hasattr(x, '__iter__') else x).astype(int).tolist()

def _get_many2one_text_serie(serie: Series) -> Series:
    """
    Les id récupérés d'odoo sont sous la forme [id, texte], on ne garde que le texte
    """
    return serie.apply(lambda x: x[1] if hasattr(x, '__iter__') else x).astype(str)