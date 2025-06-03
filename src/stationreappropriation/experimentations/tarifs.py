import marimo

__generated_with = "0.11.20"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    return mo, np, pd


@app.cell
def _(pd):
    # Création du DataFrame
    tarifs_df = pd.DataFrame({
        'Poste': ['Fonctionnement', 'Capacité et GO', 'Réseau (TURPE)', 'CTA', 'TVA'],
        'Montant (€)': [3.31, 0.06, 8.32, 1.82, 0.74]
    })
    return (tarifs_df,)


@app.cell
def _(tarifs_df):
    import matplotlib.pyplot as plt

    # Palette de couleurs "un poil plus sexy" (exemple)
    colors = ['#FF6361','#FFA600','#BC5090','#58508D','#003F5C']

    plt.figure(figsize=(4,6))
    accum = 0

    for i, row in tarifs_df.iterrows():
        plt.bar(
            'Total', 
            row['Montant (€)'], 
            bottom=accum, 
            label=row['Poste'], 
            color=colors[i % len(colors)],
            alpha=0.9
        )
        accum += row['Montant (€)']

    plt.ylabel('Montant (€)')
    plt.title('Composition de la part fixe mensuelle (Abonnement)')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
    return accum, colors, i, plt, row


@app.cell
def _(np, plt):
    # Nombre de souscriptions (ex. 1000 à 3000, pas de 100)
    N = np.arange(1000, 3001, 100)

    # On définit un prix de vente unitaire :
    PRIX_UNITAIRE = 3.31

    # Revenus de fonctionnement (supposés proportionnels au nombre de souscripteurs)
    revenus = PRIX_UNITAIRE * N


    # On veut que la "frais de fonctionnement" soit une unique droite
    # qui passe par (1000, 1200*3.31) et (1500, 1500*3.31)

    X1, Y1 = 1000, 1400 * PRIX_UNITAIRE
    X2, Y2 = 1500, 1500 * PRIX_UNITAIRE

    a = (Y2 - Y1) / (X2 - X1)  # Pente
    b = Y1 - a * X1    # Ordonnée à l'origine

    # On calcule frais_fct pour tout N entre 1000 et 3000
    frais_fct = a * N + b

    # Surplus
    surplus = revenus - frais_fct

    # Création du graphique
    plt.figure(figsize=(8, 5))

    # Tracer de la courbe des revenus
    plt.plot(N, revenus, label='Revenus', linewidth=2)

    # Tracer de la courbe des frais de fonctionnement
    plt.plot(N, frais_fct, label='Frais de fonctionnement',  linewidth=2)

    # Tracer de la courbe du surplus
    plt.plot(N, surplus, label='Surplus', linewidth=2)
    # Remplir la zone entre le surplus et l'axe (0)
    # plt.fill_between(N, surplus, 0, where=(surplus>=0), color='green', alpha=0.1, label='Surface Surplus')
    plt.fill_between(N, surplus, 0, color='green', alpha=0.1, label='Surface Surplus')

    # Marque le point d'équilibre à 1500 (optionnel)
    plt.axvline(1500, color='gray', linestyle='--', label='Équilibre théorique')

    # Tracer explicitement la ligne y=0
    plt.axhline(0, color='gray', linewidth=1, zorder=2)

    plt.xlabel('Nombre de souscriptions')
    plt.ylabel('Montant (€)')
    plt.title('Revenus, Frais de fonctionnement, Surplus')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.show()
    return N, PRIX_UNITAIRE, X1, X2, Y1, Y2, a, b, frais_fct, revenus, surplus


@app.cell
def _(a, b):
    def projection(n, a, b, r):
        revenus = r * n
        frais = a * n + b
        surplus = revenus - frais
        return revenus, frais, surplus

    r, f, s = projection(300_000, a, b, 3.31)

    r, f, s, f/300_000
    return f, projection, r, s


if __name__ == "__main__":
    app.run()
