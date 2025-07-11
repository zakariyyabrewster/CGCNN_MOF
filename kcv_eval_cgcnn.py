import os
import yaml
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

# ─── CONFIG ────────────────────────────────────────────────────────────────────
config_path = "config_kcv_cgcnn.yaml"
with open(config_path, "r") as f:
    config = yaml.full_load(f)

result_path_pre = "training_results/finetuning/CGCNN_CV"
os.makedirs("training_results/analysis", exist_ok=True)

properties = ['Di', 'Df', 'Dif', 'CH4_HP', 'CO2_LP', 'logKH_CO2']
n_folds    = config['dataloader']['num_folds']

# ─── LOAD ALL TEST RESULTS ────────────────────────────────────────────────────
def load_test_results():
    """Load test results for all properties and folds."""
    records = []
    missing_files = []
    
    for prop in properties:
        prop_folds = 0
        for fold in range(n_folds):
            file_path = (
                f"{result_path_pre}/CGCNN_fold_{fold}_{prop}"
                f"/test_results_{prop}.csv"
            )
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path, usecols=['cif_id','target','pred'])
                    df['property'] = prop
                    df['fold']     = fold
                    records.append(df)
                    prop_folds += 1
                    print(f"✓ Loaded {len(df)} samples for {prop} (fold {fold})")
                except Exception as e:
                    print(f"✗ Error loading {file_path}: {e}")
                    missing_files.append(file_path)
            else:
                print(f"✗ File not found: {file_path}")
                missing_files.append(file_path)
        
        if prop_folds < n_folds:
            print(f"⚠ WARNING: {prop} has only {prop_folds}/{n_folds} folds")
    
    if not records:
        raise FileNotFoundError("No test-result files found.")
    
    if missing_files:
        print(f"\n⚠ Missing {len(missing_files)} files total")
    
    return pd.concat(records, ignore_index=True)

# ─── CALCULATE PER-FOLD METRICS ────────────────────────────────────────────────
def calculate_fold_metrics(df):
    """Compute SRCC, MAE, MSE, RMSE, and R² per (property, fold)."""
    rows = []
    for (prop, fold), grp in df.groupby(['property','fold']):
        y_true = grp['target'].values
        y_pred = grp['pred'].values

        # Handle potential NaN values
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        if mask.sum() == 0:
            print(f"⚠ WARNING: All NaN values for {prop} fold {fold}")
            continue
            
        y_true, y_pred = y_true[mask], y_pred[mask]

        srcc, p_val = spearmanr(y_true, y_pred)
        mae         = np.mean(np.abs(y_true - y_pred))
        mse         = np.mean((y_true - y_pred)**2)
        rmse        = np.sqrt(mse)
        
        # More robust R² calculation
        ss_res = ((y_true - y_pred)**2).sum()
        ss_tot = ((y_true - y_true.mean())**2).sum()
        r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0

        rows.append({
            'property': prop,
            'fold': fold,
            'srcc': srcc,
            'srcc_pval': p_val,
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'n_samples': len(y_true)
        })
    return pd.DataFrame(rows)

# ─── ENHANCED SUMMARY WITH CONFIDENCE INTERVALS ──────────────────────────────
def create_comprehensive_summary(fold_df):
    """
    Create a comprehensive summary table with mean ± std and confidence intervals.
    """
    from scipy import stats
    
    metrics = ['srcc', 'mae', 'mse', 'rmse', 'r2']
    rows = []
    
    for prop, grp in fold_df.groupby('property'):
        row = {'Property': prop}
        
        for m in metrics:
            values = grp[m].values
            mean = np.mean(values)
            std = np.std(values, ddof=1)  # Sample standard deviation
            
            # 95% confidence interval
            if len(values) > 1:
                ci = stats.t.interval(0.95, len(values)-1, 
                                    loc=mean, scale=stats.sem(values))
                ci_str = f"[{ci[0]:.3f}, {ci[1]:.3f}]"
            else:
                ci_str = "N/A"
            
            # Format based on metric type
            if m == 'srcc' or m == 'r2':
                row[f'{m.upper()}_mean_std'] = f"{mean:.3f} ± {std:.3f}"
            else:
                row[f'{m.upper()}_mean_std'] = f"{mean:.4f} ± {std:.4f}"
            
            row[f'{m.upper()}_CI95'] = ci_str
            
        # Add sample size info
        row['Total_Samples'] = grp['n_samples'].sum()
        row['Folds_Available'] = len(grp)
        
        rows.append(row)
    
    return pd.DataFrame(rows)

# ─── STATISTICAL SIGNIFICANCE TESTING ────────────────────────────────────────
def perform_statistical_tests(fold_df):
    """
    Perform statistical tests to compare properties.
    """
    from scipy.stats import friedmanchisquare, wilcoxon
    
    print("\n=== Statistical Analysis ===")
    
    # Check if we have complete data for all properties
    pivot_srcc = fold_df.pivot(index='fold', columns='property', values='srcc')
    pivot_mae = fold_df.pivot(index='fold', columns='property', values='mae')
    
    # Friedman test for SRCC (non-parametric repeated measures)
    if pivot_srcc.notna().all().all():
        stat, p_val = friedmanchisquare(*[pivot_srcc[prop].values for prop in properties])
        print(f"Friedman test for SRCC: χ² = {stat:.4f}, p = {p_val:.4f}")
        if p_val < 0.05:
            print("✓ Significant differences detected between properties (SRCC)")
        else:
            print("✗ No significant differences between properties (SRCC)")
    
    # Friedman test for MAE
    if pivot_mae.notna().all().all():
        stat, p_val = friedmanchisquare(*[pivot_mae[prop].values for prop in properties])
        print(f"Friedman test for MAE: χ² = {stat:.4f}, p = {p_val:.4f}")
        if p_val < 0.05:
            print("✓ Significant differences detected between properties (MAE)")
        else:
            print("✗ No significant differences between properties (MAE)")

# ─── ENHANCED VISUALIZATION ─────────────────────────────────────────────────────
def create_comprehensive_plots(fold_df, data_name, out_dir="training_results/analysis"):
    """
    Create comprehensive visualization with error bars and statistical annotations.
    """
    os.makedirs(out_dir, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8')
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Main metrics comparison with error bars
    metrics = ['srcc', 'mae', 'rmse', 'r2']
    metric_titles = ['Spearman ρ (SRCC)', 'Mean Absolute Error', 'Root Mean Squared Error', 'R² Score']
    
    for i, (metric, title) in enumerate(zip(metrics, metric_titles)):
        ax = plt.subplot(2, 4, i+1)
        
        # Calculate means and stds
        summary = fold_df.groupby('property')[metric].agg(['mean', 'std']).reset_index()
        
        # Create bar plot with error bars
        bars = ax.bar(summary['property'], summary['mean'], 
                     yerr=summary['std'], capsize=5, alpha=0.7)
        
        # Add value labels on bars
        for bar, mean, std in zip(bars, summary['mean'], summary['std']):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + std,
                   f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylabel(metric.upper())
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
    
    # 2. Boxplots showing fold-to-fold variation
    for i, (metric, title) in enumerate(zip(metrics, metric_titles)):
        ax = plt.subplot(2, 4, i+5)
        
        sns.boxplot(data=fold_df, x='property', y=metric, ax=ax, showfliers=True)
        ax.set_title(f'{title} - Fold Variation', fontsize=12, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel(metric.upper())
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f"5-Fold Cross-Validation Results - {data_name}", fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    out_path = f"{out_dir}/{data_name}_comprehensive_analysis.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create a separate ranking visualization
    create_ranking_plot(fold_df, data_name, out_dir)

def create_ranking_plot(fold_df, data_name, out_dir):
    """Create a ranking visualization."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # SRCC ranking
    srcc_summary = fold_df.groupby('property')['srcc'].agg(['mean', 'std']).reset_index()
    srcc_summary = srcc_summary.sort_values('mean', ascending=False)
    
    bars1 = ax1.barh(srcc_summary['property'], srcc_summary['mean'], 
                     xerr=srcc_summary['std'], capsize=5, alpha=0.7, color='skyblue')
    ax1.set_title('Ranking by SRCC (Higher is Better)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Spearman Correlation Coefficient')
    ax1.grid(True, alpha=0.3)
    
    # Add ranking numbers
    for i, (bar, mean, std) in enumerate(zip(bars1, srcc_summary['mean'], srcc_summary['std'])):
        ax1.text(0.02, bar.get_y() + bar.get_height()/2, f'#{i+1}', 
                ha='left', va='center', fontweight='bold', fontsize=12)
    
    # MAE ranking
    mae_summary = fold_df.groupby('property')['mae'].agg(['mean', 'std']).reset_index()
    mae_summary = mae_summary.sort_values('mean', ascending=True)
    
    bars2 = ax2.barh(mae_summary['property'], mae_summary['mean'], 
                     xerr=mae_summary['std'], capsize=5, alpha=0.7, color='lightcoral')
    ax2.set_title('Ranking by MAE (Lower is Better)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Mean Absolute Error')
    ax2.grid(True, alpha=0.3)
    
    # Add ranking numbers
    for i, (bar, mean, std) in enumerate(zip(bars2, mae_summary['mean'], mae_summary['std'])):
        ax2.text(mae_summary['mean'].min() * 0.1, bar.get_y() + bar.get_height()/2, f'#{i+1}', 
                ha='left', va='center', fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    out_path = f"{out_dir}/{data_name}_ranking.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.show()

# ─── MAIN PIPELINE ────────────────────────────────────────────────────────────
def main():
    print("=== 5-Fold Cross-Validation Analysis ===\n")
    
    # 1) Load data
    all_results = load_test_results()
    print(f"\nTotal loaded samples: {len(all_results)}")
    
    # 2) Calculate per-fold metrics
    fold_metrics = calculate_fold_metrics(all_results)
    
    # Save detailed fold metrics
    fold_metrics.to_csv(
        f"training_results/analysis/{config['data_name']}_detailed_fold_metrics.csv",
        index=False
    )
    
    # 3) Create comprehensive summary
    comprehensive_summary = create_comprehensive_summary(fold_metrics)
    comprehensive_summary.to_csv(
        f"training_results/analysis/{config['data_name']}_comprehensive_summary.csv",
        index=False
    )
    
    print("\n=== COMPREHENSIVE PERFORMANCE SUMMARY ===")
    print(comprehensive_summary.to_string(index=False))
    
    # 4) Statistical tests
    perform_statistical_tests(fold_metrics)
    
    # 5) Create visualizations
    create_comprehensive_plots(fold_metrics, config['data_name'])
    
    # 6) Final ranking summary
    print("\n=== FINAL RANKINGS ===")
    srcc_ranking = fold_metrics.groupby('property')['srcc'].mean().sort_values(ascending=False)
    mae_ranking = fold_metrics.groupby('property')['mae'].mean().sort_values(ascending=True)
    
    print("\nBy SRCC (Best to Worst):")
    for i, (prop, score) in enumerate(srcc_ranking.items(), 1):
        std = fold_metrics[fold_metrics['property'] == prop]['srcc'].std()
        print(f"{i}. {prop}: {score:.3f} ± {std:.3f}")
    
    print("\nBy MAE (Best to Worst):")
    for i, (prop, score) in enumerate(mae_ranking.items(), 1):
        std = fold_metrics[fold_metrics['property'] == prop]['mae'].std()
        print(f"{i}. {prop}: {score:.4f} ± {std:.4f}")
    
    print(f"\nAll analyses complete. Check training_results/analysis/ for detailed outputs.")

if __name__ == "__main__":
    main()