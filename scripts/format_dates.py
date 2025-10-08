import pandas as pd
from datetime import datetime
import os

print("="*60)
print("🔄 UNIFORMISATION DES DATES AU FORMAT CLASSIQUE")
print("="*60)

# Créer le dossier output s'il n'existe pas
if not os.path.exists('output'):
    os.makedirs('output')
    print("\n📁 Dossier 'output' créé")

# Liste des fichiers à traiter
fichiers_input = [
    'data/Pharma_Ventes_Hourly.csv',
    'data/Pharma_Ventes_Daily.csv',
    'data/Pharma_Ventes_Weekly.csv',
    'data/Pharma_Ventes_Monthly.csv'
]

def formater_dates(fichier_input):
    """Formate les dates au format DD/MM/YYYY HH:MM:SS et sauvegarde dans output/"""
    print(f"\n📂 Traitement de : {fichier_input}")
    
    # Charger le fichier
    df = pd.read_csv(fichier_input)
    
    # Afficher le format actuel
    print(f"   Format actuel : {df['datum'].iloc[0]}")
    
    # Convertir en datetime
    df['datum'] = pd.to_datetime(df['datum'])
    
    # Formater selon le type de fichier
    if 'Hourly' in fichier_input:
        # Format avec heure : DD/MM/YYYY HH:MM:SS
        df['datum'] = df['datum'].dt.strftime('%d/%m/%Y %H:%M:%S')
    else:
        # Format sans heure : DD/MM/YYYY
        df['datum'] = df['datum'].dt.strftime('%d/%m/%Y')
    
    # Créer le nom du fichier de sortie
    nom_fichier = os.path.basename(fichier_input)
    nom_fichier_clean = nom_fichier.replace('.csv', '_CLEAN.xlsx')
    fichier_output = os.path.join('output', nom_fichier_clean)
    
    # Sauvegarder dans output/
    df.to_excel(fichier_output, index=False, engine='openpyxl')
    
    print(f"   ✅ Format nouveau : {df['datum'].iloc[0]}")
    print(f"   ✅ Fichier sauvegardé : {fichier_output}")
    
    return fichier_output

# Traiter tous les fichiers
fichiers_crees = []
for fichier in fichiers_input:
    if os.path.exists(fichier):
        fichier_sortie = formater_dates(fichier)
        fichiers_crees.append(fichier_sortie)
    else:
        print(f"\n⚠️  Fichier non trouvé : {fichier}")

print("\n" + "="*60)
print("✅ UNIFORMISATION TERMINÉE !")
print("="*60)
print("\n📋 Formats appliqués :")
print("   • Hourly : DD/MM/YYYY HH:MM:SS")
print("   • Daily  : DD/MM/YYYY")
print("   • Weekly : DD/MM/YYYY")
print("   • Monthly: DD/MM/YYYY")
print("\n📁 Fichiers créés dans output/ :")
for fichier in fichiers_crees:
    print(f"   • {fichier}")