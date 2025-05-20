import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from tkinter import font as tkfont

# Configuration des styles
STYLE = {
    'bg_color': '#E0E0E0',  # Light grey background
    'accent_color': '#B0BEC5',  # Light grey-blue border
    'text_color': '#000000',  # Black text
    'button_color_default': '#7E57C2',  # Purple for algorithm buttons
    'button_color_entree': '#66BB6A',  # Green for Entrée button
    'button_color_sortie': '#EF5350',  # Pink/Red for Sortie button
    'button_color_algos': '#42A5F5', # Blue for Algorithmes button
    'button_hover': '#9575CD',  # Lighter purple for hover
    'frame_color': '#FFFFFF',  # White frame background
    'success_color': '#4CAF50',  # Green for success messages (can keep)
    'error_color': '#f44336',  # Red for error messages (can keep)
    'text_frame_color': '#FFFFFF',  # White for text frame background
    'text_frame_border': '#B0BEC5',  # Light grey-blue for text frame border
    'title_color': '#1A237E' # Dark blue for the main title
}

# Configuration des polices
FONTS = {
    'title': ('Helvetica', 24, 'bold'),
    'subtitle': ('Helvetica', 16),
    'button': ('Helvetica', 12),
    'text': ('Helvetica', 11)
}

# Fonction pour créer un label avec cadre
def create_framed_label(parent, text, font, padding=10):
    frame = tk.Frame(
        parent,
        bg=STYLE['text_frame_color'],
        bd=2,
        relief="solid",
        highlightbackground=STYLE['text_frame_border'],
        highlightthickness=1
    )
    label = tk.Label(
        frame,
        text=text,
        font=font,
        bg=STYLE['text_frame_color'],
        fg=STYLE['text_color'],
        padx=padding,
        pady=padding
    )
    label.pack()
    return frame

# Fonction pour la régression linéaire
def regression_lineaire():
    if 'donnees_importees' not in globals():
        messagebox.showerror("Erreur", "Veuillez d'abord importer un fichier de données")
        return
        
    df = donnees_importees
    X = df.iloc[:, :-1].values  # Toutes les colonnes sauf la dernière
    y = df.iloc[:, -1].values   # Dernière colonne
    
    # Modèle
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    
    # Coefficients
    resultat = "✅ Coefficients de la régression :\n"
    for i, col in enumerate(df.columns[:-1]):
        resultat += f"{col}: {model.coef_[i]:.2f}\n"
    resultat += f"\nBiais (intercept) : {model.intercept_:.2f}\n"
    
    # Score R²
    r2 = model.score(X, y)
    resultat += f"\n✅ Score R² : {r2:.4f}\n"
    
    # Marges de variation
    resultat += "\n✅ Marges de variation :\n"
    for col in df.columns:
        resultat += f"{col}: {df[col].min():.2f} à {df[col].max():.2f}\n"
    
    # Affichage graphique
    plt.figure(figsize=(12, 6))
    
    # Graphique 1: Prédictions vs Réelles
    plt.subplot(1, 2, 1)
    plt.scatter(y, y_pred, color='blue', alpha=0.5)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    plt.title("Prédictions vs Valeurs Réelles")
    plt.xlabel("Valeurs Réelles")
    plt.ylabel("Prédictions")
    plt.grid(True)
    
    # Graphique 2: Résidus
    plt.subplot(1, 2, 2)
    residus = y - y_pred
    plt.scatter(y_pred, residus, color='green', alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title("Résidus vs Prédictions")
    plt.xlabel("Prédictions")
    plt.ylabel("Résidus")
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Afficher les résultats dans une fenêtre
    messagebox.showinfo("Résultats de la Régression Linéaire", resultat)

def clustering_kmeans():
    if 'donnees_importees' not in globals():
        messagebox.showerror("Erreur", "Veuillez d'abord importer un fichier de données")
        return
        
    df = donnees_importees
    X = df.iloc[:, :-1].values  # Toutes les colonnes sauf la dernière
    
    # Applique KMeans
    kmeans = KMeans(n_clusters=4, random_state=42)
    kmeans.fit(X)
    y_kmeans = kmeans.predict(X)
    
    # Affichage
    plt.figure(figsize=(8, 5))
    plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
                s=200, c='red', marker='x', label='Centres')
    plt.title("Clustering K-Means sur vos données")
    plt.xlabel(df.columns[0])
    plt.ylabel(df.columns[1])
    plt.legend()
    plt.grid(True)
    plt.show()

def modele_arima():
    if 'donnees_importees' not in globals():
        messagebox.showerror("Erreur", "Veuillez d'abord importer un fichier de données")
        return
        
    df = donnees_importees
    serie = df.iloc[:, -1].values  # Utilise la dernière colonne comme série temporelle
    
    # Ajustement du modèle ARIMA(p=1,d=1,q=1)
    model = ARIMA(serie, order=(1, 1, 1))
    model_fit = model.fit()
    
    # Prédiction
    forecast = model_fit.predict(start=0, end=len(serie)-1)
    
    # Affichage
    plt.figure(figsize=(9, 5))
    plt.plot(range(len(serie)), serie, label="Données réelles", color="blue")
    plt.plot(range(len(forecast)), forecast, label="Prévision ARIMA", color="red")
    plt.title("Prévision ARIMA sur vos données")
    plt.xlabel("Temps")
    plt.ylabel(df.columns[-1])
    plt.legend()
    plt.grid(True)
    plt.show()

def random_forest_iris():
    if 'donnees_importees' not in globals():
        messagebox.showerror("Erreur", "Veuillez d'abord importer un fichier de données")
        return
        
    df = donnees_importees
    X = df.iloc[:, :-1].values  # Toutes les colonnes sauf la dernière
    y = df.iloc[:, -1].values   # Dernière colonne
    
    # Réduction à 2 dimensions pour affichage
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # Entraîner le modèle
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    y_pred = model.predict(X)
    
    # Affichage
    plt.figure(figsize=(8, 5))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_pred, cmap='viridis')
    plt.title("Classification Random Forest sur vos données")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.grid(True)
    plt.show()

def validation_croisee():
    if 'donnees_importees' not in globals():
        messagebox.showerror("Erreur", "Veuillez d'abord importer un fichier de données")
        return
        
    df = donnees_importees
    X = df.iloc[:, :-1].values  # Toutes les colonnes sauf la dernière
    y = df.iloc[:, -1].values   # Dernière colonne
    
    # Modèle
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Validation croisée à 5 plis
    scores = cross_val_score(model, X, y, cv=5)
    
    # Affichage des scores
    print("✅ Scores validation croisée (5 plis) :", scores)
    print(f"🎯 Moyenne de précision : {np.mean(scores):.4f}")
    
    # Affichage graphique des scores
    plt.figure(figsize=(7, 4))
    plt.plot(range(1, 6), scores, marker='o', linestyle='-', label='Score par pli')
    plt.axhline(np.mean(scores), color='red', linestyle='--', label='Moyenne')
    plt.title("Validation croisée sur vos données")
    plt.xlabel("Pli (Fold)")
    plt.ylabel("Score")
    plt.xticks(range(1, 6))
    plt.legend()
    plt.grid(True)
    plt.show()

def selectionner_fichier():
    # Ouvrir la boîte de dialogue pour sélectionner un fichier CSV
    fichier = filedialog.askopenfilename(
        title="Sélectionner un fichier CSV",
        filetypes=[("Fichiers CSV", "*.csv"), ("Tous les fichiers", "*.*")]
    )
    
    if fichier:
        try:
            # Lire le fichier CSV
            df = pd.read_csv(fichier)
            
            # Vérifier le format des données
            if len(df.columns) != 4:
                raise ValueError("Le fichier doit contenir exactement 4 colonnes")
            
            if len(df) < 10:
                raise ValueError("Le fichier doit contenir au moins 10 lignes de données")
            
            # Stocker les données pour une utilisation ultérieure
            global donnees_importees
            donnees_importees = df
            
            # Afficher un message de confirmation avec les statistiques
            stats = f"Fichier chargé avec succès!\n\n"
            stats += f"Nombre de lignes: {len(df)}\n"
            stats += f"Colonnes: {', '.join(df.columns)}\n\n"
            stats += "Statistiques des données:\n"
            stats += df.describe().to_string()
            
            messagebox.showinfo("Succès", stats)
            
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors de la lecture du fichier: {str(e)}")


# Création de la fenêtre principale
fenetre = tk.Tk()
fenetre.title("Interface Graphique IA")
fenetre.geometry("800x600")
fenetre.configure(bg=STYLE['bg_color'])

# Style pour les boutons
def create_button(parent, text, command, width=20, button_type='default'):
    btn = tk.Button(
        parent,
        text=text,
        command=command,
        font=FONTS['button'],
        bg=STYLE[f'button_color_{button_type}'],
        fg=STYLE['text_color'],
        activebackground=STYLE['button_hover'],
        activeforeground=STYLE['text_color'],
        relief=tk.FLAT,
        width=width,
        cursor="hand2"
    )
    return btn

# Fonction pour afficher les algorithmes
def afficher_algorithmes():
    cadre_algos.pack(pady=20)

# Titre avec style luxe
titre_frame = tk.Frame(fenetre, bg=STYLE['bg_color'])
titre_frame.pack(pady=20)

titre = create_framed_label(
    titre_frame,
    "Interface Graphique IA",
    FONTS['title'],
    padding=15
)
titre.pack(pady=5)

titre_label = titre.winfo_children()[0]
titre_label.config(fg=STYLE['title_color'])

# Cadre principal avec style
cadre_principal = tk.Frame(
    fenetre,
    bg=STYLE['frame_color'],
    bd=2,
    relief="solid",
    highlightbackground=STYLE['text_frame_border'],
    highlightthickness=1,
    padx=30,
    pady=20
)
cadre_principal.pack(pady=20, padx=20, fill="both", expand=True)

# Bouton principal avec style
btn_algos = create_button(
    cadre_principal,
    "Algorithmes de L'Intelligence Artificielle",
    afficher_algorithmes,
    width=30,
    button_type='algos'
)
btn_algos.pack(pady=10)

# Boutons Entrée / Sortie avec style
btn_frame = tk.Frame(
    cadre_principal,
    bg=STYLE['frame_color'],
    bd=2,
    relief="solid",
    highlightbackground=STYLE['text_frame_border'],
    highlightthickness=1,
    padx=10,
    pady=10
)
btn_frame.pack(pady=10)

btn_entree = create_button(btn_frame, "Entrée", selectionner_fichier, button_type='entree')
btn_entree.pack(side="left", padx=10)

btn_sortie = create_button(btn_frame, "Sortie", fenetre.quit, button_type='sortie')
btn_sortie.pack(side="right", padx=10)

# Cadre des algorithmes avec style
cadre_algos = tk.Frame(
    fenetre,
    bg=STYLE['frame_color'],
    bd=2,
    relief="solid",
    highlightbackground=STYLE['text_frame_border'],
    highlightthickness=1,
    padx=20,
    pady=20
)

# Liste des algorithmes avec style
algos = [
    "Régression Linéaire",
    "Clustering",
    "TimeSeries ARIMA",
    "Random Forest",
    "Validation Croisée"
]

# Dictionnaire de mapping pour les noms d'algorithmes
algo_mapping = {
    "Régression Linéaire": "regression_lineaire",
    "Clustering": "clustering_kmeans",
    "TimeSeries ARIMA": "modele_arima",
    "Random Forest": "random_forest_iris",
    "Validation Croisée": "validation_croisee"
}

for algo in algos:
    btn = create_button(
        cadre_algos,
        algo,
        lambda a=algo: globals()[algo_mapping[a]](),
        button_type='default'
    )
    btn.pack(pady=5)

# Configuration du style des graphiques
plt.style.use('dark_background')

# Fonction pour personnaliser les graphiques
def style_graphique():
    plt.rcParams['figure.facecolor'] = STYLE['bg_color']
    plt.rcParams['axes.facecolor'] = STYLE['frame_color']
    plt.rcParams['axes.edgecolor'] = STYLE['accent_color']
    plt.rcParams['axes.labelcolor'] = STYLE['text_color']
    plt.rcParams['xtick.color'] = STYLE['text_color']
    plt.rcParams['ytick.color'] = STYLE['text_color']
    plt.rcParams['grid.color'] = STYLE['accent_color']
    plt.rcParams['grid.alpha'] = 0.3

# Appliquer le style aux graphiques
style_graphique()

# Lancer l'application
fenetre.mainloop()
