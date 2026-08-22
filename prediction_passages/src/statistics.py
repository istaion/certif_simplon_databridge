"""
Visualisation des résultats de l'ensemble learning.
Inclut analyse détaillée par site.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pathlib import Path


# Configuration du style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.size'] = 10


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calcule les métriques pour une série, en ignorant les NaN dans y_pred."""
    valid = ~np.isnan(y_pred) & ~np.isnan(y_true)
    if valid.sum() == 0:
        return {'MAE': np.nan, 'RMSE': np.nan, 'MAPE': np.nan, 'Biais': np.nan, 'N': 0}

    yt, yp = y_true[valid], y_pred[valid]
    mae = mean_absolute_error(yt, yp)
    rmse = np.sqrt(mean_squared_error(yt, yp))

    mask = yt != 0
    mape = float(np.mean(np.abs((yt[mask] - yp[mask]) / yt[mask])) * 100) if mask.sum() > 0 else np.nan
    bias = float(np.mean(yp - yt))

    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'Biais': bias, 'N': int(valid.sum())}


def get_metrics_by_site(results_df: pd.DataFrame) -> pd.DataFrame:
    """Calcule les métriques par site pour chaque modèle."""
    models = ['XGBoost', 'Prophet', 'ARIMA', 'MovingAverage', 'Ensemble']
    # Accepte aussi les variantes Prophet21/35, XGBoost21/35 issues du pivot Trino
    available_models = []
    for m in models:
        if f'pred_{m}' in results_df.columns:
            available_models.append(m)
        else:
            # Cherche les variantes numérotées (ex: Prophet21, XGBoost21)
            for col in results_df.columns:
                if col.startswith(f'pred_{m}') and col not in [f'pred_{x}' for x in available_models]:
                    available_models.append(col.replace('pred_', ''))

    sites = results_df['login_site'].unique()
    all_metrics = []

    for site in sites:
        site_df = results_df[results_df['login_site'] == site]
        y_true = site_df['efreel'].values

        site_metrics = {
            'login_site': site,
            'N_obs': len(site_df),
            'Moyenne_reelle': float(np.nanmean(y_true)),
            'Std_reelle': float(np.nanstd(y_true)),
        }

        for model in available_models:
            col = f'pred_{model}'
            if col in site_df.columns:
                y_pred = site_df[col].values
                metrics = calculate_metrics(y_true, y_pred)
                site_metrics[f'{model}_MAE'] = metrics['MAE']
                site_metrics[f'{model}_RMSE'] = metrics['RMSE']
                site_metrics[f'{model}_MAPE'] = metrics['MAPE']
                site_metrics[f'{model}_Biais'] = metrics['Biais']

        all_metrics.append(site_metrics)

    return pd.DataFrame(all_metrics).sort_values('Moyenne_reelle', ascending=False)


def plot_site_predictions(results_df: pd.DataFrame, output_dir: str = "."):
    """Génère un graphique par site avec toutes les prédictions."""
    sites = results_df['login_site'].unique()

    # Détecter les modèles disponibles depuis les colonnes pred_*
    models = [c.replace('pred_', '') for c in results_df.columns if c.startswith('pred_')]

    palette = [
        '#e74c3c', '#3498db', '#2ecc71', '#9b59b6',
        '#f39c12', '#1abc9c', '#e67e22', '#8e44ad',
    ]
    color_cycle = {m: palette[i % len(palette)] for i, m in enumerate(models)}
    color_cycle['Réel'] = '#2c3e50'

    n_sites = len(sites)
    n_cols = 3
    n_rows = (n_sites + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
    fig.suptitle("Prédictions vs Réel par Site", fontsize=16, fontweight='bold', y=1.02)

    if n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    for idx, site in enumerate(sorted(sites)):
        ax = axes_flat[idx]
        site_df = results_df[results_df['login_site'] == site].sort_values('efdate')
        y_true = site_df['efreel'].values

        ax.plot(site_df['efdate'], site_df['efreel'],
                color=color_cycle['Réel'], linewidth=2, label='Réel', marker='o', markersize=3)

        for model in models:
            col = f'pred_{model}'
            if col not in site_df.columns:
                continue
            is_ensemble = 'ensemble' in model.lower()
            ax.plot(site_df['efdate'], site_df[col],
                    color=color_cycle[model],
                    linewidth=2 if is_ensemble else 1.2,
                    linestyle='-' if is_ensemble else '--',
                    label=model, alpha=0.8)

        # Titre : MAE Ensemble + MAE du premier modèle non-Ensemble disponible
        title_parts = [f"Moy={np.nanmean(y_true):.0f}"]
        for model in models:
            col = f'pred_{model}'
            if col not in site_df.columns:
                continue
            y_pred = site_df[col].values
            valid = ~np.isnan(y_pred) & ~np.isnan(y_true)
            if valid.sum() == 0:
                continue
            mae = mean_absolute_error(y_true[valid], y_pred[valid])
            title_parts.append(f"{model}={mae:.1f}")

        ax.set_title(f"Site {site}\n" + ", ".join(title_parts), fontsize=8)
        ax.tick_params(axis='x', rotation=45)
        ax.set_xlabel('')
        ax.set_ylabel('Effectif')

        if idx == 0:
            ax.legend(loc='upper left', fontsize=7)

    for idx in range(len(sites), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/predictions_par_site.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Graphique sauvegardé: {output_dir}/predictions_par_site.png")


def plot_metrics_heatmap(metrics_df: pd.DataFrame, output_dir: str = "."):
    """Heatmap des MAE par site et modèle."""
    mae_cols = [c for c in metrics_df.columns if c.endswith('_MAE')]
    models = [c.replace('_MAE', '') for c in mae_cols]

    heatmap_data = metrics_df[['login_site'] + mae_cols].set_index('login_site')
    heatmap_data.columns = models
    
    # Trier par moyenne réelle (du plus gros au plus petit site)
    site_order = metrics_df.sort_values('Moyenne_reelle', ascending=False)['login_site']
    heatmap_data = heatmap_data.loc[site_order]
    
    fig, ax = plt.subplots(figsize=(12, max(8, len(site_order) * 0.4)))
    
    sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='RdYlGn_r',
                ax=ax, cbar_kws={'label': 'MAE'})
    
    ax.set_title("MAE par Site et Modèle\n(Sites triés par volume décroissant)", fontsize=14)
    ax.set_xlabel("Modèle")
    ax.set_ylabel("Site")
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/mae_heatmap_sites.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Heatmap sauvegardé: {output_dir}/mae_heatmap_sites.png")


def plot_mape_vs_volume(metrics_df: pd.DataFrame, output_dir: str = "."):
    """Scatter plot MAPE vs Volume moyen par site."""
    mape_cols = [c for c in metrics_df.columns if c.endswith('_MAPE')]
    models = [c.replace('_MAPE', '') for c in mape_cols]
    palette = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#e67e22']
    colors = palette[:len(models)]

    fig, ax = plt.subplots(figsize=(12, 8))

    for model, color in zip(models, colors):
        mape_col = f'{model}_MAPE'
        ax.scatter(metrics_df['Moyenne_reelle'], metrics_df[mape_col],
                   s=100, alpha=0.7, label=model, c=color, edgecolors='black')

    ax.set_xlabel("Volume moyen du site", fontsize=12)
    ax.set_ylabel("MAPE (%)", fontsize=12)
    ax.set_title("MAPE vs Volume moyen par site\n(Les petits sites ont souvent un MAPE plus élevé)", fontsize=14)
    ax.legend()
    max_mape = metrics_df[mape_cols].max().max()
    ax.set_ylim(0, min(500, max_mape * 1.1) if not np.isnan(max_mape) else 500)

    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='MAPE 50%')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/mape_vs_volume.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Scatter plot sauvegardé: {output_dir}/mape_vs_volume.png")


def plot_best_model_by_site(metrics_df: pd.DataFrame, output_dir: str = "."):
    """Barplot montrant le meilleur modèle par site."""
    mae_cols = [c for c in metrics_df.columns if c.endswith('_MAE')]
    
    # Trouver le meilleur modèle par site
    metrics_df = metrics_df.copy()
    metrics_df['Meilleur_modele'] = metrics_df[mae_cols].idxmin(axis=1).str.replace('_MAE', '')
    metrics_df['Meilleur_MAE'] = metrics_df[mae_cols].min(axis=1)
    
    # Compter les victoires
    model_wins = metrics_df['Meilleur_modele'].value_counts()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Graphique 1: Nombre de sites où chaque modèle est le meilleur
    colors_map = {
        'XGBoost': '#e74c3c',
        'Prophet': '#3498db',
        'ARIMA': '#2ecc71',
        'MovingAverage': '#9b59b6',
        'Ensemble': '#f39c12'
    }
    bar_colors = [colors_map.get(m, 'gray') for m in model_wins.index]
    
    axes[0].bar(model_wins.index, model_wins.values, color=bar_colors, edgecolor='black')
    axes[0].set_title("Nombre de sites où chaque modèle est le meilleur", fontsize=12)
    axes[0].set_xlabel("Modèle")
    axes[0].set_ylabel("Nombre de sites")
    
    for i, (model, wins) in enumerate(model_wins.items()):
        axes[0].text(i, wins + 0.2, str(wins), ha='center', fontsize=12, fontweight='bold')
    
    # Graphique 2: MAE du meilleur modèle par site
    site_order = metrics_df.sort_values('Moyenne_reelle', ascending=False)
    bar_colors_site = [colors_map.get(m, 'gray') for m in site_order['Meilleur_modele']]
    
    axes[1].barh(range(len(site_order)), site_order['Meilleur_MAE'], color=bar_colors_site, edgecolor='black')
    axes[1].set_yticks(range(len(site_order)))
    axes[1].set_yticklabels([f"{s} ({m})" for s, m in zip(site_order['login_site'], site_order['Meilleur_modele'])])
    axes[1].set_xlabel("MAE")
    axes[1].set_title("MAE du meilleur modèle par site\n(trié par volume décroissant)", fontsize=12)
    axes[1].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/meilleur_modele_par_site.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Best model plot sauvegardé: {output_dir}/meilleur_modele_par_site.png")


def plot_error_by_weekday_and_site(results_df: pd.DataFrame, output_dir: str = "."):
    """Analyse des erreurs par jour de semaine."""
    days = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi']
    models = [c.replace('pred_', '') for c in results_df.columns if c.startswith('pred_')]

    results_df = results_df.copy()

    # Boxplot sur le premier modèle disponible (Ensemble en priorité, sinon le premier)
    primary = next((m for m in models if 'ensemble' in m.lower()), models[0]) if models else None

    if primary:
        results_df['_error_primary'] = np.abs(
            results_df['efreel'] - results_df[f'pred_{primary}']
        )

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    weekday_data = results_df[results_df['day'] < 5]

    if primary:
        box_data = [
            weekday_data[weekday_data['day'] == d]['_error_primary'].dropna().values
            for d in range(5)
        ]
        bp = axes[0].boxplot(box_data, labels=days, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('#f39c12')
            patch.set_alpha(0.7)
        axes[0].set_title(f"Distribution des erreurs ({primary}) par jour", fontsize=12)
    else:
        axes[0].set_title("Aucun modèle disponible", fontsize=12)

    axes[0].set_ylabel("Erreur absolue")
    axes[0].set_xlabel("Jour de la semaine")

    # MAE par jour pour chaque modèle
    mae_by_day = []
    for day in range(5):
        day_data = weekday_data[weekday_data['day'] == day]
        if len(day_data) == 0:
            continue
        for model in models:
            col = f'pred_{model}'
            y_pred = day_data[col].values
            y_true = day_data['efreel'].values
            valid = ~np.isnan(y_pred) & ~np.isnan(y_true)
            if valid.sum() == 0:
                continue
            mae = mean_absolute_error(y_true[valid], y_pred[valid])
            mae_by_day.append({'Jour': days[day], 'Modèle': model, 'MAE': mae})
    
    mae_df = pd.DataFrame(mae_by_day)
    mae_pivot = mae_df.pivot(index='Jour', columns='Modèle', values='MAE')
    mae_pivot = mae_pivot.loc[days]  # Réordonner
    
    mae_pivot.plot(kind='bar', ax=axes[1], width=0.8, edgecolor='black')
    axes[1].set_title("MAE par jour de semaine et modèle", fontsize=12)
    axes[1].set_xlabel("Jour")
    axes[1].set_ylabel("MAE")
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].legend(title='Modèle')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/erreur_par_jour.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Error by weekday sauvegardé: {output_dir}/erreur_par_jour.png")


def generate_site_report(metrics_df: pd.DataFrame, output_dir: str = "."):
    """Génère un rapport CSV détaillé par site."""
    models = [c.replace('_MAE', '') for c in metrics_df.columns if c.endswith('_MAE')]

    cols_order = ['login_site', 'N_obs', 'Moyenne_reelle', 'Std_reelle']
    for model in models:
        cols_order.extend([f'{model}_MAE', f'{model}_MAPE', f'{model}_Biais'])

    cols_order = [c for c in cols_order if c in metrics_df.columns]
    
    report_df = metrics_df[cols_order].copy()
    
    # Arrondir
    numeric_cols = report_df.select_dtypes(include=[np.number]).columns
    report_df[numeric_cols] = report_df[numeric_cols].round(2)
    
    # Sauvegarder
    report_df.to_csv(f"{output_dir}/metriques_par_site.csv", index=False)
    print(f"Rapport CSV sauvegardé: {output_dir}/metriques_par_site.csv")
    
    return report_df


def plot_mase_boxplot_by_model(results_df: pd.DataFrame, output_dir: str = ".", naive_column: str = 'pred_MovingAverage'):
    """Génère des boîtes à moustaches du MASE pour chaque modèle.

    Le MASE est calculé par site : MAE(modèle) / MAE(naïf).
    Par défaut, le forecast naïf est pred_MovingAverage (moyenne mobile).
    Un MASE < 1 signifie que le modèle bat la moyenne mobile.
    """
    # Si naive_column est pred_MovingAverage, on l'exclut des modèles comparés
    # (son MASE serait toujours 1 par construction)
    all_models = ['XGBoost', 'Prophet', 'ARIMA', 'MovingAverage', 'Ensemble']
    naive_model = naive_column.replace('pred_', '') if naive_column.startswith('pred_') else None
    models = [m for m in all_models if m != naive_model]

    colors_map = {
        'XGBoost': '#e74c3c',
        'Prophet': '#3498db',
        'ARIMA': '#2ecc71',
        'MovingAverage': '#9b59b6',
        'Ensemble': '#f39c12'
    }

    # Calculer le MASE par site et par modèle
    sites = results_df['login_site'].unique()
    mase_data = {model: [] for model in models}

    for site in sites:
        site_df = results_df[results_df['login_site'] == site]
        y_true = site_df['efreel'].values

        if naive_column not in site_df.columns:
            continue
        y_naive = site_df[naive_column].values

        # Masque pour ignorer les NaN dans le naïf
        valid = ~np.isnan(y_naive) & ~np.isnan(y_true)
        if valid.sum() == 0:
            continue

        mae_naive = mean_absolute_error(y_true[valid], y_naive[valid])
        if mae_naive == 0:
            continue

        for model in models:
            col = f'pred_{model}'
            if col not in site_df.columns:
                continue
            y_pred = site_df[col].values
            valid_m = valid & ~np.isnan(y_pred)
            if valid_m.sum() == 0:
                continue
            mae_model = mean_absolute_error(y_true[valid_m], y_pred[valid_m])
            mase_data[model].append(mae_model / mae_naive)

    # Créer le graphique
    fig, ax = plt.subplots(figsize=(12, 8))

    box_data = [mase_data[model] for model in models]
    positions = list(range(1, len(models) + 1))

    bp = ax.boxplot(box_data, positions=positions, labels=models,
                    patch_artist=True, widths=0.6,
                    showmeans=True, meanline=False,
                    boxprops=dict(linewidth=1.5),
                    medianprops=dict(color='black', linewidth=2),
                    meanprops=dict(marker='D', markerfacecolor='red',
                                  markeredgecolor='red', markersize=8))

    for patch, model in zip(bp['boxes'], models):
        patch.set_facecolor(colors_map[model])
        patch.set_alpha(0.7)

    naive_label = naive_model if naive_model else naive_column
    ax.axhline(y=1, color='gray', linestyle='--', linewidth=2, alpha=0.7,
               label=f'MASE = 1 (baseline : {naive_label})')

    # Annotations médiane / moyenne au-dessus de chaque boîte
    y_max = max((max(d) for d in box_data if d), default=2)
    ax.set_ylim(bottom=0, top=y_max * 1.18)
    for i, model in enumerate(models, 1):
        data = mase_data[model]
        if data:
            median = np.median(data)
            mean = np.mean(data)
            ax.text(i, y_max * 1.12,
                    f'Med: {median:.2f}\nMoy: {mean:.2f}',
                    ha='center', va='top', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel('Modèle', fontsize=12, fontweight='bold')
    ax.set_ylabel('MASE', fontsize=12, fontweight='bold')
    ax.set_title(
        f'Distribution du MASE par modèle\n'
        f'(naïf = {naive_label} | MASE < 1 ⟹ meilleur que la moyenne mobile)',
        fontsize=14, fontweight='bold'
    )
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/mase_boxplot_modeles.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Boxplot MASE sauvegardé: {output_dir}/mase_boxplot_modeles.png")


def print_summary_report(metrics_df: pd.DataFrame, results_df: pd.DataFrame):
    """Affiche un résumé dans la console."""
    mae_cols = [c for c in metrics_df.columns if c.endswith('_MAE')]
    models = [c.replace('_MAE', '') for c in mae_cols]

    print("\n" + "=" * 80)
    print("RAPPORT DÉTAILLÉ PAR SITE")
    print("=" * 80)

    print("\n1. MÉTRIQUES PAR SITE (triés par volume décroissant)")
    print("-" * 80)

    for _, row in metrics_df.iterrows():
        print(f"\n  Site: {row['login_site']}")
        print(f"    Observations: {row['N_obs']}, Moyenne: {row['Moyenne_reelle']:.1f}, Std: {row['Std_reelle']:.1f}")
        print(f"    {'Modèle':<20} {'MAE':>10} {'MAPE':>10} {'Biais':>10}")
        print(f"    {'-'*50}")

        best_mae = float('inf')
        best_model = ''

        for model in models:
            mae = row.get(f'{model}_MAE', np.nan)
            mape = row.get(f'{model}_MAPE', np.nan)
            biais = row.get(f'{model}_Biais', np.nan)

            if not np.isnan(mae) and mae < best_mae:
                best_mae = mae
                best_model = model

            marker = " *" if model == best_model else ""
            mae_s = f"{mae:>10.1f}" if not np.isnan(mae) else f"{'—':>10}"
            mape_s = f"{mape:>9.1f}%" if not np.isnan(mape) else f"{'—':>9} "
            biais_s = f"{biais:>+10.1f}" if not np.isnan(biais) else f"{'—':>10}"
            print(f"    {model:<20} {mae_s} {mape_s} {biais_s}{marker}")

    # Résumé global
    print("\n" + "=" * 80)
    print("RÉSUMÉ GLOBAL")
    print("=" * 80)

    wins = metrics_df[mae_cols].idxmin(axis=1).str.replace('_MAE', '', regex=False).value_counts()
    print("\n  Nombre de sites où chaque modèle est le meilleur:")
    for model in models:
        print(f"    {model}: {wins.get(model, 0)} sites")

    # Sites problématiques (MAPE > 100%) — utilise Ensemble s'il existe
    ensemble_mape_col = 'Ensemble_MAPE'
    if ensemble_mape_col in metrics_df.columns:
        print("\n  Sites avec MAPE Ensemble > 100%:")
        problematic = metrics_df[metrics_df[ensemble_mape_col] > 100]
        for _, row in problematic.iterrows():
            print(f"    {row['login_site']}: MAPE={row[ensemble_mape_col]:.1f}%, Moyenne={row['Moyenne_reelle']:.1f}")