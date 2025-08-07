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

# Paths for results
cgcnn_path_pre = "training_results/finetuning/CGCNN_CV"
transformer_path_pre = "training_results/finetuning/Transformer_CV"
pn_path_pre = "training_results/finetuning/PointNet_CV"
os.makedirs("training_results/cross_model_analysis", exist_ok=True)

properties = ['Di', 'Df', 'Dif', 'CH4_HP', 'CO2_LP', 'logKH_CO2']
n_folds = config['dataloader']['num_folds']

# ─── LOAD RESULTS FOR ALL THREE MODELS ────────────────────────────────────────
def load_model_results(model_name, path_prefix):
    """Load test results for a specific model across all properties and folds."""
    records = []
    missing_files = []
    
    for prop in properties:
        prop_folds = 0
        for fold in range(n_folds):
            if model_name == 'CGCNN':
                file_path = f"{path_prefix}/CGCNN_fold_{fold}_{prop}/test_results_{prop}.csv"
            if model_name == 'Transformer':  # Transformer
                file_path = f"{path_prefix}/Trans_fold_{fold}_{prop}/test_results_{prop}.csv"
            if model_name == 'PointNet':  # PointNet
                file_path = f"{path_prefix}/PointNet_fold_{fold}_{prop}/test_results_{prop}.csv"

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
    """Load results for all three models."""
    print("=== Loading CGCNN Results ===")
    cgcnn_results = load_model_results('CGCNN', cgcnn_path_pre)
    
    print("\n=== Loading Transformer Results ===")
    transformer_results = load_model_results('Transformer', transformer_path_pre)
    
    print("\n=== Loading PointNet Results ===")
    pn_results = load_model_results('PointNet', pn_path_pre)

    # Combine results
    if not cgcnn_results.empty and not transformer_results.empty and not pn_results.empty:
        all_results = pd.concat([cgcnn_results, transformer_results, pn_results], ignore_index=True)
        return all_results
    else:
        raise ValueError("Could not load results for one or more models")

# ─── CALCULATE METRICS FOR ALL THREE MODELS ───────────────────────────────────
def calculate_cross_model_metrics(df):
    """Calculate metrics for all three models across all properties and folds."""
    results = []

    for model in ['CGCNN', 'Transformer', 'PointNet']:
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
    """Perform descriptive statistical comparison across all three models."""
    print("\n=== Statistical Comparison (Descriptive)")
    
    comparison_results = []
    test_metrics = ['srcc', 'mae', 'mse', 'r2']
    
    for prop in properties:
        prop_data = metrics_df[metrics_df['property'] == prop]
        
        cgcnn_data = prop_data[prop_data['model'] == 'CGCNN']
        transformer_data = prop_data[prop_data['model'] == 'Transformer']
        pn_data = prop_data[prop_data['model'] == 'PointNet']
        
        # Check if we have data for all models
        if len(cgcnn_data) == 0 or len(transformer_data) == 0 or len(pn_data) == 0:
            print(f"⚠ Skipping {prop} - missing data for one or more models")
            continue
            
        prop_results = {'property': prop}
        
        for metric in test_metrics:
            cgcnn_values = cgcnn_data[metric].values
            transformer_values = transformer_data[metric].values
            pn_values = pn_data[metric].values
            
            # Calculate descriptive statistics for each model
            cgcnn_mean = np.mean(cgcnn_values)
            transformer_mean = np.mean(transformer_values)
            pn_mean = np.mean(pn_values)
            
            cgcnn_std = np.std(cgcnn_values, ddof=1) if len(cgcnn_values) > 1 else 0
            transformer_std = np.std(transformer_values, ddof=1) if len(transformer_values) > 1 else 0
            pn_std = np.std(pn_values, ddof=1) if len(pn_values) > 1 else 0
            
            # Determine best model for this metric and property
            if metric in ['srcc', 'r2']:  # Higher is better
                means = {'CGCNN': cgcnn_mean, 'Transformer': transformer_mean, 'PointNet': pn_mean}
                best_model = max(means, key=means.get)
                best_value = means[best_model]
            else:  # Lower is better (mae, mse)
                means = {'CGCNN': cgcnn_mean, 'Transformer': transformer_mean, 'PointNet': pn_mean}
                best_model = min(means, key=means.get)
                best_value = means[best_model]
            
            prop_results[f'{metric}_cgcnn_mean'] = cgcnn_mean
            prop_results[f'{metric}_cgcnn_std'] = cgcnn_std
            prop_results[f'{metric}_transformer_mean'] = transformer_mean
            prop_results[f'{metric}_transformer_std'] = transformer_std
            prop_results[f'{metric}_pointnet_mean'] = pn_mean
            prop_results[f'{metric}_pointnet_std'] = pn_std
            prop_results[f'{metric}_best_model'] = best_model
            prop_results[f'{metric}_best_value'] = best_value
            
            print(f"{prop} - {metric.upper()}: {best_model} best ({best_value:.4f})")
        
        comparison_results.append(prop_results)
    
    return pd.DataFrame(comparison_results)

# ─── CREATE COMPREHENSIVE COMPARISON TABLE ─────────────────────────────────────
# ─── CREATE COMPREHENSIVE COMPARISON TABLE ─────────────────────────────────────
def create_comparison_summary(metrics_df):
    """Create a comprehensive comparison table."""
    summary_data = []
    
    for prop in properties:
        prop_data = metrics_df[metrics_df['property'] == prop]
        
        for model in ['CGCNN', 'Transformer', 'PointNet']:
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
    
    # Define consistent color mapping for all models
    model_colors = {
        'CGCNN': 'skyblue',
        'Transformer': 'lightcoral', 
        'PointNet': 'lightgreen'
    }
    
    # Set up the figure
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    axes = axes.flatten()
    
    metrics = ['srcc', 'mae', 'mse', 'r2']
    metric_titles = ['Spearman ρ (SRCC)', 'Mean Absolute Error', 'Mean Squared Error', 'R² Score']
    
    # Create ordered categorical for consistent plotting
    metrics_df['property'] = pd.Categorical(metrics_df['property'], categories=properties, ordered=True)
    
    for i, (metric, title) in enumerate(zip(metrics, metric_titles)):
        # Bar plot comparison
        ax1 = axes[i]
        
        # Calculate means for all three models
        summary = metrics_df.groupby(['property', 'model'])[metric].agg(['mean', 'std']).reset_index()
        
        # Create grouped bar plot
        x = np.arange(len(properties))
        width = 0.25  # Narrower bars for three models
        
        cgcnn_data = summary[summary['model'] == 'CGCNN']
        transformer_data = summary[summary['model'] == 'Transformer']
        pointnet_data = summary[summary['model'] == 'PointNet']
        
        # Ensure data is properly ordered
        cgcnn_data = cgcnn_data.set_index('property').reindex(properties).reset_index()
        transformer_data = transformer_data.set_index('property').reindex(properties).reset_index()
        pointnet_data = pointnet_data.set_index('property').reindex(properties).reset_index()
        
        bars1 = ax1.bar(x - width, cgcnn_data['mean'], width, 
                       yerr=cgcnn_data['std'], capsize=3, 
                       label='CGCNN', alpha=0.8, color=model_colors['CGCNN'])
        bars2 = ax1.bar(x, transformer_data['mean'], width,
                       yerr=transformer_data['std'], capsize=3,
                       label='Transformer', alpha=0.8, color=model_colors['Transformer'])
        bars3 = ax1.bar(x + width, pointnet_data['mean'], width,
                       yerr=pointnet_data['std'], capsize=3,
                       label='PointNet', alpha=0.8, color=model_colors['PointNet'])
        
        ax1.set_xlabel('Property')
        ax1.set_ylabel(metric.upper())
        ax1.set_title(f'{title} Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(properties, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Box plot comparison
        ax2 = axes[i + 4]
        sns.boxplot(data=metrics_df, x='property', y=metric, hue='model', 
                   palette=model_colors, ax=ax2)
        ax2.set_title(f'{title} - Distribution Comparison')
        ax2.set_xlabel('Property')
        ax2.set_ylabel(metric.upper())
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
    
    plt.suptitle('CGCNN vs Transformer vs PointNet - Performance Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    plt.savefig(f"{out_dir}/model_comparison_comprehensive.png", dpi=300, bbox_inches='tight')
    plt.show()


# ─── MAIN PIPELINE ────────────────────────────────────────────────────────────
def main():
    print("=== Cross-Model Evaluation ===\n")
    
    # 1) Load data for all three models
    try:
        all_results = load_all_results()
        print(f"\nTotal loaded samples: {len(all_results)}")
        print(f"Models: {all_results['model'].unique()}")
        print(f"Properties: {all_results['property'].unique()}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # 2) Calculate metrics for all three models
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
        pointnet_avg = prop_metrics[prop_metrics['model'] == 'PointNet']['srcc'].mean()
        
        # Find the best model for this property
        model_scores = {'CGCNN': cgcnn_avg, 'Transformer': transformer_avg, 'PointNet': pointnet_avg}
        best_model = max(model_scores, key=model_scores.get)
        
        print(f"{prop}: {best_model} (CGCNN: {cgcnn_avg:.3f}, Transformer: {transformer_avg:.3f}, PointNet: {pointnet_avg:.3f})")
    
    print(f"\nAll analyses complete. Check training_results/cross_model_analysis/ for detailed outputs.")

if __name__ == "__main__":
    main()