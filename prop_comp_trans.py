'''
SRCC on test_results to determine effects of
geometric vs chemistry dependent properties
on CGCNN Performance'''


# COPILOT DRAFT
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os
import yaml

config = yaml.load(open("config_ft_transformer.yaml", "r"), Loader=yaml.FullLoader)

result_path_pre = f'training_results/finetuning/Transformer/Trans_{config['fine_tune_from']}_{config['dataset']['data_name']}_{config['dataloader']['randomSeed']}_'

os.makedirs('training_results/analysis', exist_ok=True)

def load_test_results():
    """Load test results for all n properties"""
    properties = ['Di', 'Df', 'Dif', 'CH4_HP', 'CO2_LP', 'logKH_CO2'] # edit depending on properties to compare
    all_results = []

    for prop in properties:
        file_path = f'{result_path_pre}{prop}/test_results_{prop}.csv'

        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df['property'] = prop
            all_results.append(df)
            print(f"Loaded {len(df)} samples for {prop}")
        else:
            print(f"File not found: {file_path}")

    if all_results:
        combined_results = pd.concat(all_results, ignore_index=True)
        return combined_results
    else:
        raise FileNotFoundError("No test result files found")

def calculate_performance_metrics(test_results):
    """Calculate performance metrics for each property"""
    results = []

    for prop in test_results['property'].unique():
        prop_data = test_results[test_results['property'] == prop]

        # Calculate metrics
        true_vals = prop_data['target']
        pred_vals = prop_data['pred']

        # SRCC (Spearman Rank Correlation Coefficient)
        srcc, p_value = spearmanr(true_vals, pred_vals)

        # MAE and MSE
        mae = np.mean(np.abs(true_vals - pred_vals))
        mse = np.mean((true_vals - pred_vals) ** 2)
        rmse = np.sqrt(mse)

        # R-squared
        ss_res = np.sum((true_vals - pred_vals) ** 2)
        ss_tot = np.sum((true_vals - np.mean(true_vals)) ** 2)
        r2 = 1 - (ss_res / ss_tot)

        results.append({
            'property': prop,
            'srcc': srcc,
            'p_value': p_value,
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'n_samples': len(prop_data)
        })

    return pd.DataFrame(results)

def rank_model_performance(performance_df):
    """Rank model performance by different metrics"""

    print("=== Model Performance Summary ===")
    print(performance_df.round(4))

    # Rank by SRCC (higher is better)
    print("\n=== Ranking by SRCC (Best to Worst) ===")
    srcc_ranking = performance_df.sort_values('srcc', ascending=False)
    for i, row in enumerate(srcc_ranking.itertuples(), 1):
        print(f"{i}. {row.property}: SRCC = {row.srcc:.4f}, MAE = {row.mae:.4f}")

    # Rank by MAE (lower is better)
    print("\n=== Ranking by MAE (Best to Worst) ===")
    mae_ranking = performance_df.sort_values('mae', ascending=True)
    for i, row in enumerate(mae_ranking.itertuples(), 1):
        print(f"{i}. {row.property}: MAE = {row.mae:.4f}, SRCC = {row.srcc:.4f}")

    return performance_df

def visualize_results(test_results, performance_df):
    fig = plt.figure(figsize=(24, 12))
    gs = gridspec.GridSpec(2, 12, height_ratios=[1, 1.6], hspace=0.5, wspace=1)

    metrics = ['srcc', 'mae', 'mse', 'r2']
    metric_titles = ['Spearman Correlation (SRCC)', 'Mean Absolute Error (MAE)', 
                     'Mean Squared Error (MSE)', 'R² Score']
    
    properties = test_results['property'].unique()
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold', 'orchid', 'lightgray']

    # Top row: 4 plots, each spans 3 columns (3×4 = 12)
    for i, (metric, title) in enumerate(zip(metrics, metric_titles)):
        ax = fig.add_subplot(gs[0, i*3:(i+1)*3])
        ax.bar(performance_df['property'], performance_df[metric], color=colors[:len(performance_df)])
        ax.set_title(title, fontsize=13)
        ax.set_ylabel(metric.upper())
        ax.tick_params(axis='x', rotation=45)

    # Bottom row: 6 scatter plots, each spans 2 columns (2×6 = 12)
    for i, prop in enumerate(properties):
        ax = fig.add_subplot(gs[1, i*2:(i+1)*2])
        prop_data = test_results[test_results['property'] == prop]
        ax.scatter(prop_data['target'], prop_data['pred'], alpha=0.6, color=colors[i])
        min_val = min(prop_data['target'].min(), prop_data['pred'].min())
        max_val = max(prop_data['target'].max(), prop_data['pred'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        ax.set_xlabel('True')
        ax.set_ylabel('Predicted')
        srcc_val = performance_df[performance_df['property'] == prop]['srcc'].iloc[0]
        ax.set_title(f'{prop} (SRCC: {srcc_val:.3f})', fontsize=11)

    plt.savefig(f'training_results/analysis/{config['dataset']['data_name']}_trans_{len(properties)}properties_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


def create_detailed_comparison(performance_df):
    """Create detailed comparison table"""

    # Normalize metrics for comparison (0-1 scale)
    comparison_df = performance_df.copy()

    # For SRCC and R²: higher is better (keep as is)
    comparison_df['srcc_norm'] = comparison_df['srcc']
    comparison_df['r2_norm'] = comparison_df['r2']

    # For MAE, MSE, RMSE: lower is better (invert)
    comparison_df['mae_norm'] = 1 - (comparison_df['mae'] / comparison_df['mae'].max())
    comparison_df['mse_norm'] = 1 - (comparison_df['mse'] / comparison_df['mse'].max())
    comparison_df['rmse_norm'] = 1 - (comparison_df['rmse'] / comparison_df['rmse'].max())

    # Calculate overall performance score
    comparison_df['overall_score'] = (comparison_df['srcc_norm'] +
                                    comparison_df['r2_norm'] +
                                    comparison_df['mae_norm']) / 3

    print("\n=== Overall Performance Ranking ===")
    overall_ranking = comparison_df.sort_values('overall_score', ascending=False)
    for i, row in enumerate(overall_ranking.itertuples(), 1):
        print(f"{i}. {row.property}: Overall Score = {row.overall_score:.4f}")

    return comparison_df

def main():
    """Main analysis function"""
    print("Loading test results for 4 properties...")
    test_results = load_test_results()

    print("\nCalculating performance metrics...")
    performance_df = calculate_performance_metrics(test_results)

    print("\nRanking model performance...")
    ranked_results = rank_model_performance(performance_df)

    print("\nCreating detailed comparison...")
    comparison_df = create_detailed_comparison(performance_df)

    print("\nCreating visualizations...")
    visualize_results(test_results, performance_df)

    # Save results
    performance_df.to_csv(f'training_results/analysis/{config['dataset']['data_name']}_trans_{len(test_results['property'].unique())}properties_performance.csv', index=False)
    comparison_df.to_csv(f'training_results/analysis/{config['dataset']['data_name']}_trans_{len(test_results['property'].unique())}6properties_comparison.csv', index=False)
    print("\nResults saved to CSV files")

    return performance_df, comparison_df

if __name__ == "__main__":
    performance_results, comparison_results = main()