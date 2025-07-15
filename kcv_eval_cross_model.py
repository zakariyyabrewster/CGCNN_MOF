import os
import yaml
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, wilcoxon, mannwhitneyu
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ─── CONFIG ────────────────────────────────────────────────────────────────────
config_path = "config_kcv_cgcnn.yaml"
with open(config_path, "r") as f:
    config = yaml.full_load(f)

# Paths for results
cgcnn_path_pre = "training_results/finetuning/CGCNN_CV"
transformer_path_pre = "training_results/finetuning/Transformer_CV"
os.makedirs("training_results/cross_model_analysis", exist_ok=True)

properties = ['Di', 'Df', 'Dif', 'CH4_HP', 'CO2_LP', 'logKH_CO2']
n_folds = config['dataloader']['num_folds']

# ─── LOAD RESULTS FOR BOTH MODELS ──────────────────────────────────────────────
def load_model_results(model_name, path_prefix):
    """Load test results for a specific model across all properties and folds."""
    records = []
    missing_files = []
    
    for prop in properties:
        prop_folds = 0
        for fold in range(n_folds):
            if model_name == 'CGCNN':
                file_path = f"{path_prefix}/CGCNN_fold_{fold}_{prop}/test_results_{prop}.csv"
            else:  # Transformer
                file_path = f"{path_prefix}/Trans_fold_{fold}_{prop}/test_results_{prop}.csv"
            
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path, usecols=['cif_id','target','pred'])
                    df['property'] = prop
                    df['fold'] = fold
                    df['model'] = model_name
                    records.append(df)
                    prop_folds += 1
                    print(f"✓ {model_name} - Loaded {len(df)} samples for {prop} (fold {fold})")
                except Exception as e:
                    print(f"✗ {model_name} - Error loading {file_path}: {e}")
                    missing_files.append(file_path)
            else:
                print(f"✗ {model_name} - File not found: {file_path}")
                missing_files.append(file_path)
        
        if prop_folds < n_folds:
            print(f"⚠ {model_name} - WARNING: {prop} has only {prop_folds}/{n_folds} folds")
    
    if not records:
        print(f"No test-result files found for {model_name}")
        return pd.DataFrame()
    
    if missing_files:
        print(f"\n⚠ {model_name} - Missing {len(missing_files)} files total")
    
    return pd.concat(records, ignore_index=True)

def load_all_results():
    """Load results for both models."""
    print("=== Loading CGCNN Results ===")
    cgcnn_results = load_model_results('CGCNN', cgcnn_path_pre)
    
    print("\n=== Loading Transformer Results ===")
    transformer_results = load_model_results('Transformer', transformer_path_pre)
    
    # Combine results
    if not cgcnn_results.empty and not transformer_results.empty:
        all_results = pd.concat([cgcnn_results, transformer_results], ignore_index=True)
        return all_results
    else:
        raise ValueError("Could not load results for one or both models")

# ─── CALCULATE METRICS FOR BOTH MODELS ─────────────────────────────────────────
def calculate_cross_model_metrics(df):
    """Calculate metrics for both models across all properties and folds."""
    results = []
    
    for model in ['CGCNN', 'Transformer']:
        model_data = df[df['model'] == model]
        
        for (prop, fold), grp in model_data.groupby(['property', 'fold']):
            y_true = grp['target'].values
            y_pred = grp['pred'].values
            
            # Handle potential NaN values
            mask = ~(np.isnan(y_true) | np.isnan(y_pred))
            if mask.sum() == 0:
                print(f"⚠ WARNING: All NaN values for {model} - {prop} fold {fold}")
                continue
                
            y_true, y_pred = y_true[mask], y_pred[mask]
            
            srcc, p_val = spearmanr(y_true, y_pred)
            mae = np.mean(np.abs(y_true - y_pred))
            mse = np.mean((y_true - y_pred)**2)
            rmse = np.sqrt(mse)
            
            # R² calculation
            ss_res = ((y_true - y_pred)**2).sum()
            ss_tot = ((y_true - y_true.mean())**2).sum()
            r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0
            
            results.append({
                'model': model,
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
    
    return pd.DataFrame(results)

# ─── STATISTICAL COMPARISON ────────────────────────────────────────────────────
def perform_statistical_comparison(metrics_df):
    """Perform statistical tests comparing models."""
    print("\n=== Statistical Comparison: CGCNN vs Transformer ===")
    
    comparison_results = []
    test_metrics = ['srcc', 'mae', 'mse', 'r2']
    
    for prop in properties:
        prop_data = metrics_df[metrics_df['property'] == prop]
        
        cgcnn_data = prop_data[prop_data['model'] == 'CGCNN']
        transformer_data = prop_data[prop_data['model'] == 'Transformer']
        
        # Check if we have data for both models
        if len(cgcnn_data) == 0 or len(transformer_data) == 0:
            print(f"⚠ Skipping {prop} - missing data for one or both models")
            continue
            
        prop_results = {'property': prop}
        
        for metric in test_metrics:
            cgcnn_values = cgcnn_data[metric].values
            transformer_values = transformer_data[metric].values
            
            if len(cgcnn_values) > 0 and len(transformer_values) > 0:
                # Paired t-test (if same folds) or Mann-Whitney U test
                if len(cgcnn_values) == len(transformer_values):
                    # Paired comparison
                    stat, p_val = wilcoxon(cgcnn_values, transformer_values)
                    test_type = "Wilcoxon"
                else:
                    # Independent samples
                    stat, p_val = mannwhitneyu(cgcnn_values, transformer_values)
                    test_type = "Mann-Whitney U"
                
                # Calculate effect size (Cohen's d)
                pooled_std = np.sqrt(((len(cgcnn_values)-1)*np.var(cgcnn_values, ddof=1) + 
                                    (len(transformer_values)-1)*np.var(transformer_values, ddof=1)) / 
                                   (len(cgcnn_values) + len(transformer_values) - 2))
                cohens_d = (np.mean(cgcnn_values) - np.mean(transformer_values)) / pooled_std
                
                prop_results[f'{metric}_stat'] = stat
                prop_results[f'{metric}_pval'] = p_val
                prop_results[f'{metric}_cohens_d'] = cohens_d
                prop_results[f'{metric}_test'] = test_type
                
                # Determine winner
                if metric in ['srcc', 'r2']:  # Higher is better
                    winner = 'CGCNN' if np.mean(cgcnn_values) > np.mean(transformer_values) else 'Transformer'
                else:  # Lower is better (mae, mse)
                    winner = 'CGCNN' if np.mean(cgcnn_values) < np.mean(transformer_values) else 'Transformer'
                
                prop_results[f'{metric}_winner'] = winner
                prop_results[f'{metric}_significant'] = p_val < 0.05
                
                print(f"{prop} - {metric.upper()}: {winner} wins (p={p_val:.4f}, d={cohens_d:.3f})")
        
        comparison_results.append(prop_results)
    
    return pd.DataFrame(comparison_results)

# ─── CREATE COMPREHENSIVE COMPARISON TABLE ─────────────────────────────────────
def create_comparison_summary(metrics_df):
    """Create a comprehensive comparison table."""
    summary_data = []
    
    for prop in properties:
        prop_data = metrics_df[metrics_df['property'] == prop]
        
        for model in ['CGCNN', 'Transformer']:
            model_data = prop_data[prop_data['model'] == model]
            
            if len(model_data) == 0:
                continue
                
            row = {
                'Property': prop,
                'Model': model,
                'SRCC_mean': model_data['srcc'].mean(),
                'SRCC_std': model_data['srcc'].std(),
                'MAE_mean': model_data['mae'].mean(),
                'MAE_std': model_data['mae'].std(),
                'MSE_mean': model_data['mse'].mean(),
                'MSE_std': model_data['mse'].std(),
                'R2_mean': model_data['r2'].mean(),
                'R2_std': model_data['r2'].std(),
                'n_folds': len(model_data)
            }
            summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    return summary_df

# ─── VISUALIZATION ─────────────────────────────────────────────────────────────
def create_comparison_plots(metrics_df, out_dir="training_results/cross_model_analysis"):
    """Create comprehensive comparison visualizations."""
    os.makedirs(out_dir, exist_ok=True)
    
    plt.style.use('seaborn-v0_8')
    
    # Set up the figure
    fig, axes = plt.subplots(2, 4, figsize=(20, 12))
    axes = axes.flatten()
    
    metrics = ['srcc', 'mae', 'mse', 'r2']
    metric_titles = ['Spearman ρ (SRCC)', 'Mean Absolute Error', 'Mean Squared Error', 'R² Score']
    
    # Create ordered categorical for consistent plotting
    metrics_df['property'] = pd.Categorical(metrics_df['property'], categories=properties, ordered=True)
    
    for i, (metric, title) in enumerate(zip(metrics, metric_titles)):
        # Bar plot comparison
        ax1 = axes[i]
        
        # Calculate means for both models
        summary = metrics_df.groupby(['property', 'model'])[metric].agg(['mean', 'std']).reset_index()
        
        # Create grouped bar plot
        x = np.arange(len(properties))
        width = 0.35
        
        cgcnn_data = summary[summary['model'] == 'CGCNN']
        transformer_data = summary[summary['model'] == 'Transformer']
        
        # Ensure data is properly ordered
        cgcnn_data = cgcnn_data.set_index('property').reindex(properties).reset_index()
        transformer_data = transformer_data.set_index('property').reindex(properties).reset_index()
        
        bars1 = ax1.bar(x - width/2, cgcnn_data['mean'], width, 
                       yerr=cgcnn_data['std'], capsize=5, 
                       label='CGCNN', alpha=0.8, color='skyblue')
        bars2 = ax1.bar(x + width/2, transformer_data['mean'], width,
                       yerr=transformer_data['std'], capsize=5,
                       label='Transformer', alpha=0.8, color='lightcoral')
        
        ax1.set_xlabel('Property')
        ax1.set_ylabel(metric.upper())
        ax1.set_title(f'{title} Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(properties, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Box plot comparison
        ax2 = axes[i + 4]
        sns.boxplot(data=metrics_df, x='property', y=metric, hue='model', ax=ax2)
        ax2.set_title(f'{title} - Distribution Comparison')
        ax2.set_xlabel('Property')
        ax2.set_ylabel(metric.upper())
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
    
    plt.suptitle('CGCNN vs Transformer - Performance Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    plt.savefig(f"{out_dir}/model_comparison_comprehensive.png", dpi=300, bbox_inches='tight')
    plt.show()


# ─── MAIN PIPELINE ────────────────────────────────────────────────────────────
def main():
    print("=== Cross-Model Evaluation: CGCNN vs Transformer ===\n")
    
    # 1) Load data for both models
    try:
        all_results = load_all_results()
        print(f"\nTotal loaded samples: {len(all_results)}")
        print(f"Models: {all_results['model'].unique()}")
        print(f"Properties: {all_results['property'].unique()}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # 2) Calculate metrics for both models
    print("\n=== Calculating Performance Metrics ===")
    metrics_df = calculate_cross_model_metrics(all_results)
    
    # Save detailed metrics
    metrics_df.to_csv("training_results/cross_model_analysis/detailed_metrics_comparison.csv", index=False)
    
    # 3) Create summary comparison table
    print("\n=== Creating Comparison Summary ===")
    summary_df = create_comparison_summary(metrics_df)
    summary_df.to_csv("training_results/cross_model_analysis/summary_comparison.csv", index=False)
    
    print("\n=== PERFORMANCE SUMMARY ===")
    print(summary_df.round(4))
    
    # 4) Statistical comparison
    print("\n=== Performing Statistical Tests ===")
    comparison_df = perform_statistical_comparison(metrics_df)
    comparison_df.to_csv("training_results/cross_model_analysis/statistical_comparison.csv", index=False)
    
    # 5) Create visualizations
    print("\n=== Creating Visualizations ===")
    create_comparison_plots(metrics_df)


    
    # Property-wise analysis
    print("\n=== Property-wise Performance ===")
    for prop in properties:
        prop_metrics = metrics_df[metrics_df['property'] == prop]
        
        cgcnn_avg = prop_metrics[prop_metrics['model'] == 'CGCNN']['srcc'].mean()
        transformer_avg = prop_metrics[prop_metrics['model'] == 'Transformer']['srcc'].mean()
        
        better_model = 'CGCNN' if cgcnn_avg > transformer_avg else 'Transformer'
        print(f"{prop}: {better_model} (CGCNN: {cgcnn_avg:.3f}, Transformer: {transformer_avg:.3f})")
    
    print(f"\nAll analyses complete. Check training_results/cross_model_analysis/ for detailed outputs.")

if __name__ == "__main__":
    main()