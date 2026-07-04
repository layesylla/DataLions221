import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


class DataProfiler:
    """Profiler les données et générer des rapports."""
    
    def __init__(self, data_path="data/raw"):
        self.data_path = Path(data_path)
        self.train_path = self.data_path / "train.csv"
        self.test_path = self.data_path / "test.csv"
        self.profile_report = {}
        
    def load_data(self):
        """Charge les données"""
        print("=" * 80)
        print("📊 PROFILAGE DES DONNÉES")
        print("=" * 80)
        
        self.train = pd.read_csv(self.train_path)
        self.test = pd.read_csv(self.test_path)
        
        print(f"\n✅ Train chargé : {self.train.shape[0]} lignes, {self.train.shape[1]} colonnes")
        print(f"✅ Test chargé  : {self.test.shape[0]} lignes, {self.test.shape[1]} colonnes")
        
        return self.train, self.test
    
    def detect_data_types(self, df, name):
        """Détecte les types de données"""
        types = {
            'numerical': [],
            'categorical': [],
            'binary': [],
            'date': [],
            'text': [],
            'id': []
        }
        
        for col in df.columns:
            # Détection des IDs
            if 'id' in col.lower() or col.lower() == 'id':
                types['id'].append(col)
                continue
            
            # Détection des dates
            try:
                pd.to_datetime(df[col], errors='raise')
                types['date'].append(col)
                continue
            except:
                pass
            
            # Détection des types
            if df[col].dtype in ['int64', 'float64']:
                n_unique = df[col].nunique()
                if n_unique <= 2:
                    types['binary'].append(col)
                else:
                    types['numerical'].append(col)
            else:
                n_unique = df[col].nunique()
                if n_unique <= 2:
                    types['binary'].append(col)
                elif n_unique <= 50:
                    types['categorical'].append(col)
                else:
                    types['text'].append(col)
        
        return types
    
    def analyze_numerical(self, df, cols, name):
        """Analyse les colonnes numériques"""
        if not cols:
            return {}
        
        stats = {}
        for col in cols:
            stats[col] = {
                'count': int(df[col].count()),
                'nulls': int(df[col].isnull().sum()),
                'null_pct': float(df[col].isnull().sum() / len(df) * 100),
                'mean': float(df[col].mean()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                'q25': float(df[col].quantile(0.25)),
                'median': float(df[col].median()),
                'q75': float(df[col].quantile(0.75)),
                'max': float(df[col].max()),
                'skew': float(df[col].skew()),
                'kurtosis': float(df[col].kurtosis()),
                'outliers_count': int(self.detect_outliers(df[col])),
                'unique_values': int(df[col].nunique())
            }
        
        return stats
    
    def detect_outliers(self, series):
        """Détecte les outliers avec la méthode IQR"""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return ((series < lower_bound) | (series > upper_bound)).sum()
    
    def analyze_categorical(self, df, cols, name):
        """Analyse les colonnes catégorielles"""
        if not cols:
            return {}
        
        stats = {}
        for col in cols:
            value_counts = df[col].value_counts()
            stats[col] = {
                'count': int(df[col].count()),
                'nulls': int(df[col].isnull().sum()),
                'null_pct': float(df[col].isnull().sum() / len(df) * 100),
                'unique_values': int(df[col].nunique()),
                'top_values': value_counts.head(10).to_dict(),
                'cardinality': int(df[col].nunique()),
                'most_frequent': str(value_counts.index[0]) if len(value_counts) > 0 else None,
                'freq_pct': float(value_counts.iloc[0] / len(df) * 100) if len(value_counts) > 0 else 0
            }
        
        return stats
    
    def analyze_correlations(self):
        """Analyse les corrélations entre les features"""
        # Ne prendre que les colonnes numériques
        num_cols = self.train.select_dtypes(include=[np.number]).columns.tolist()
        
        # Exclure les IDs et les colonnes à faible variance
        exclude = ['id'] + [col for col in num_cols if self.train[col].nunique() <= 2]
        num_cols = [col for col in num_cols if col not in exclude]
        
        if len(num_cols) > 1:
            corr_matrix = self.train[num_cols].corr()
            
            # Trouver les corrélations fortes avec la cible
            target = 'status' if 'status' in self.train.columns else None
            if target and target in corr_matrix.columns:
                target_corr = corr_matrix[target].sort_values(ascending=False)
                strong_corr = target_corr[abs(target_corr) > 0.1]
            else:
                target_corr = None
                strong_corr = None
            
            return {
                'matrix': corr_matrix.to_dict(),
                'target_correlation': target_corr.to_dict() if target_corr is not None else None,
                'strong_correlations': strong_corr.to_dict() if strong_corr is not None else None,
                'high_corr_features': self.find_high_correlations(corr_matrix)
            }
        return {}
    
    def find_high_correlations(self, corr_matrix, threshold=0.8):
        """Trouve les paires de features fortement corrélées"""
        high_corr = []
        cols = corr_matrix.columns.tolist()
        for i in range(len(cols)):
            for j in range(i+1, len(cols)):
                if abs(corr_matrix.iloc[i, j]) > threshold:
                    high_corr.append({
                        'feature1': cols[i],
                        'feature2': cols[j],
                        'correlation': float(corr_matrix.iloc[i, j])
                    })
        return high_corr
    
    def analyze_missing_values(self, df, name):
        """Analyse les valeurs manquantes"""
        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100)
        
        missing_df = pd.DataFrame({
            'column': missing.index,
            'missing_count': missing.values,
            'missing_percentage': missing_pct.values
        }).sort_values('missing_percentage', ascending=False)
        
        return missing_df[missing_df['missing_count'] > 0].to_dict('records')
    
    def detect_skewness(self, df, cols):
        """Détecte les colonnes avec une forte asymétrie"""
        skewed = {}
        for col in cols:
            skew_val = df[col].skew()
            if abs(skew_val) > 1:
                skewed[col] = {
                    'skewness': float(skew_val),
                    'severity': 'high' if abs(skew_val) > 2 else 'moderate',
                    'suggested_transform': self.suggest_transform(skew_val)
                }
        return skewed
    
    def suggest_transform(self, skew_val):
        """Suggère une transformation basée sur l'asymétrie"""
        if skew_val > 2:
            return 'log1p'
        elif skew_val > 1:
            return 'sqrt'
        elif skew_val < -2:
            return 'exp'
        elif skew_val < -1:
            return 'square'
        else:
            return 'none'
    
    def detect_binary(self, df, cols):
        """Analyse les colonnes binaires"""
        binary_info = {}
        for col in cols:
            values = df[col].value_counts()
            binary_info[col] = {
                'values': values.index.tolist(),
                'counts': values.tolist(),
                'balance_ratio': float(values.min() / values.max()) if len(values) == 2 else None
            }
        return binary_info
    
    def generate_full_report(self):
        """Génère un rapport complet"""
        print("\n" + "=" * 80)
        print("🔍 GÉNÉRATION DU RAPPORT DE PROFILAGE")
        print("=" * 80)
        
        # Charger les données
        self.load_data()
        
        # Identifier la colonne cible
        target_col = 'status' if 'status' in self.train.columns else None
        
        # Types de données
        train_types = self.detect_data_types(self.train, 'Train')
        test_types = self.detect_data_types(self.test, 'Test')
        
        print("\n📋 TYPES DE DONNÉES - TRAIN:")
        for type_name, cols in train_types.items():
            if cols:
                print(f"  {type_name}: {len(cols)} colonnes - {cols[:5]}{'...' if len(cols) > 5 else ''}")
        
        # Analyses
        report = {
            'metadata': {
                'generated_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'train_shape': self.train.shape,
                'test_shape': self.test.shape,
                'target_column': target_col
            },
            'train_types': train_types,
            'test_types': test_types,
            'train_missing': self.analyze_missing_values(self.train, 'Train'),
            'test_missing': self.analyze_missing_values(self.test, 'Test'),
            'numerical_train': self.analyze_numerical(self.train, train_types['numerical'], 'Train'),
            'categorical_train': self.analyze_categorical(self.train, train_types['categorical'], 'Train'),
            'binary_train': self.detect_binary(self.train, train_types['binary']),
            'skewed_features': self.detect_skewness(self.train, train_types['numerical']),
            'correlations': self.analyze_correlations()
        }
        
        self.profile_report = report
        
        # Sauvegarder le rapport
        self.save_report(report)
        
        # Afficher un résumé
        self.display_summary(report)
        
        return report
    
    def save_report(self, report):
        """Sauvegarde le rapport en JSON et en texte"""
        # Dossier pour les rapports
        report_dir = Path("reports")
        report_dir.mkdir(exist_ok=True)
        
        # Sauvegarde JSON
        json_path = report_dir / "profiling_report.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n💾 Rapport JSON sauvegardé: {json_path}")
        
        # Sauvegarde texte lisible
        txt_path = report_dir / "profiling_report.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("📊 RAPPORT DE PROFILAGE DES DONNÉES\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Généré le: {report['metadata']['generated_at']}\n")
            f.write(f"Train shape: {report['metadata']['train_shape']}\n")
            f.write(f"Test shape: {report['metadata']['test_shape']}\n")
            f.write(f"Target: {report['metadata']['target_column']}\n\n")
            
            f.write("-" * 80 + "\n")
            f.write("VALEURS MANQUANTES\n")
            f.write("-" * 80 + "\n")
            if report['train_missing']:
                f.write("\nTRAIN:\n")
                for miss in report['train_missing']:
                    f.write(f"  {miss['column']}: {miss['missing_count']} ({miss['missing_percentage']:.2f}%)\n")
            if report['test_missing']:
                f.write("\nTEST:\n")
                for miss in report['test_missing']:
                    f.write(f"  {miss['column']}: {miss['missing_count']} ({miss['missing_percentage']:.2f}%)\n")
            
            f.write("\n" + "-" * 80 + "\n")
            f.write("COLONNES NUMÉRIQUES - STATISTIQUES\n")
            f.write("-" * 80 + "\n")
            for col, stats in report['numerical_train'].items():
                f.write(f"\n{col}:\n")
                f.write(f"  Mean: {stats['mean']:.2f}, Std: {stats['std']:.2f}\n")
                f.write(f"  Min: {stats['min']:.2f}, Q25: {stats['q25']:.2f}, Median: {stats['median']:.2f}, Q75: {stats['q75']:.2f}, Max: {stats['max']:.2f}\n")
                f.write(f"  Skew: {stats['skew']:.2f}, Outliers: {stats['outliers_count']}\n")
            
            f.write("\n" + "-" * 80 + "\n")
            f.write("COLONNES CATÉGORIELLES\n")
            f.write("-" * 80 + "\n")
            for col, stats in report['categorical_train'].items():
                f.write(f"\n{col}:\n")
                f.write(f"  Unique values: {stats['unique_values']}\n")
                f.write(f"  Most frequent: {stats['most_frequent']} ({stats['freq_pct']:.2f}%)\n")
                f.write(f"  Top 5: {list(stats['top_values'].items())[:5]}\n")
            
            if report.get('correlations'):
                f.write("\n" + "-" * 80 + "\n")
                f.write("CORRÉLATIONS\n")
                f.write("-" * 80 + "\n")
                if report['correlations'].get('target_correlation'):
                    f.write("\nCorrélation avec la cible:\n")
                    for feat, corr in report['correlations']['target_correlation'].items():
                        f.write(f"  {feat}: {corr:.3f}\n")
                if report['correlations'].get('high_corr_features'):
                    f.write("\nPaires fortement corrélées (>0.8):\n")
                    for pair in report['correlations']['high_corr_features']:
                        f.write(f"  {pair['feature1']} - {pair['feature2']}: {pair['correlation']:.3f}\n")
        
        print(f"💾 Rapport texte sauvegardé: {txt_path}")
    
    def display_summary(self, report):
        """Affiche un résumé du rapport"""
        print("\n" + "=" * 80)
        print("📊 RÉSUMÉ DU PROFILAGE")
        print("=" * 80)
        
        print(f"\n📈 Shape: Train={report['metadata']['train_shape']}, Test={report['metadata']['test_shape']}")
        
        if report['train_missing']:
            print(f"\n⚠️  Valeurs manquantes - Train: {len(report['train_missing'])} colonnes")
            for miss in report['train_missing'][:5]:
                print(f"  - {miss['column']}: {miss['missing_percentage']:.1f}%")
        
        if report['test_missing']:
            print(f"\n⚠️  Valeurs manquantes - Test: {len(report['test_missing'])} colonnes")
        
        if report.get('correlations', {}).get('target_correlation'):
            print("\n🔥 Corrélations avec la cible:")
            corr = report['correlations']['target_correlation']
            for feat, val in list(corr.items())[:5]:
                print(f"  - {feat}: {val:.3f}")
        
        if report.get('correlations', {}).get('high_corr_features'):
            print(f"\n🔗 Paires fortement corrélées: {len(report['correlations']['high_corr_features'])}")
        
        print("\n" + "=" * 80)
        print("✅ Profilage terminé ! Les rapports sont dans le dossier 'reports/'")
        print("=" * 80)

    def get_claude_prompt(self):
        """Génère un prompt optimisé pour Claude"""
        report = self.profile_report
        
        prompt = f"""
# CONTEXTE - PRÉDICTION DE STATUT

## 📊 DONNÉES
- Train: {report['metadata']['train_shape'][0]} lignes, {report['metadata']['train_shape'][1]} colonnes
- Test: {report['metadata']['test_shape'][0]} lignes, {report['metadata']['test_shape'][1]} colonnes
- Target: {report['metadata']['target_column']}

## 🔍 TYPES DE DONNÉES
**Train:**
"""
        for type_name, cols in report['train_types'].items():
            if cols:
                prompt += f"\n- {type_name}: {', '.join(cols[:10])}"
                if len(cols) > 10:
                    prompt += f" et {len(cols)-10} autres"

        prompt += f"""

## ⚠️ VALEURS MANQUANTES
**Train:**
"""
        for miss in report['train_missing']:
            prompt += f"\n- {miss['column']}: {miss['missing_percentage']:.1f}%"

        prompt += f"""

**Test:**
"""
        for miss in report['test_missing']:
            prompt += f"\n- {miss['column']}: {miss['missing_percentage']:.1f}%"

        if report.get('correlations', {}).get('target_correlation'):
            prompt += f"""

## 🔥 CORRÉLATIONS AVEC LA CIBLE
"""
            for feat, corr in report['correlations']['target_correlation'].items():
                if abs(corr) > 0.1:
                    prompt += f"\n- {feat}: {corr:.3f}"

        if report.get('correlations', {}).get('high_corr_features'):
            prompt += f"""

## 🔗 PAIRES FORTEMENT CORRÉLÉES
"""
            for pair in report['correlations']['high_corr_features']:
                prompt += f"\n- {pair['feature1']} ↔ {pair['feature2']}: {pair['correlation']:.3f}"

        prompt += """
## 💡 RECOMMANDATIONS POUR LE MODÈLE

1. **Traitement des valeurs manquantes**:
   - [À déterminer selon les colonnes]

2. **Feature Engineering**:
   - [Basé sur les corrélations et la distribution]

3. **Modèles suggérés**:
   - LightGBM, XGBoost, CatBoost
   - Stacking/Ensemble

4. **Validation**:
   - Stratified K-Fold (classification déséquilibrée)
   - Métrique à optimiser: [À définir]

5. **Feature Importance**:
   - Analyser les features les plus importantes
   - Vérifier la stabilité du modèle
"""
        
        return prompt

def main():
    """Fonction principale"""
    profiler = DataProfiler()
    report = profiler.generate_full_report()
    
    # Générer le prompt pour Claude
    prompt = profiler.get_claude_prompt()
    
    # Sauvegarder le prompt
    with open("reports/claude_prompt.txt", "w", encoding="utf-8") as f:
        f.write(prompt)
    print("\n📝 Prompt pour Claude sauvegardé: reports/claude_prompt.txt")
    
    print("\n" + "=" * 80)
    print("✅ FIN DU PROFILAGE")
    print("=" * 80)

if __name__ == "__main__":
    main()