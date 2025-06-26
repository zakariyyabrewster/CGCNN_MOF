''' 
SRCC on test_results to determine effects of 
geometric vs chemistry dependent properties 
on CGCNN Performance'''


# COPILOT DRAFT
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs('training_results/analysis', exist_ok=True)

def load_test_results():
    """Load test results for all 4 properties"""
    properties = ['Di', 'CH4_HP', 'CO2_LP', 'logKH_CO2']
    all_results = []
    
    for prop in properties:
        file_path = f'training_results/finetuning/CGCNN/CGCNN_scratch_CoRE2019_1_{prop}/test_results_{prop}.csv'
        
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
    """Create visualizations for the analysis"""
    
    fig, axes = plt.subplots(2, 4, figsize=(18, 12))
    
    # Plot 1: SRCC comparison
    axes[0,0].bar(performance_df['property'], performance_df['srcc'], 
                  color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
    axes[0,0].set_title('Spearman Correlation by Property')
    axes[0,0].set_ylabel('SRCC')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # Plot 2: MAE comparison
    axes[0,1].bar(performance_df['property'], performance_df['mae'],
                  color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
    axes[0,1].set_title('Mean Absolute Error by Property')
    axes[0,1].set_ylabel('MAE')
    axes[0,1].tick_params(axis='x', rotation=45)

    # Plot 3: MSE comparison
    axes[0,2].bar(performance_df['property'], performance_df['mse'],
                  color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
    axes[0,2].set_title('Mean Squared Error by Property')
    axes[0,2].set_ylabel('MSE')
    axes[0,2].tick_params(axis='x', rotation=45)
    
    # Plot 3: R² comparison
    axes[0,3].bar(performance_df['property'], performance_df['r2'],
                  color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
    axes[0,3].set_title('R² Score by Property')
    axes[0,3].set_ylabel('R²')
    axes[0,3].tick_params(axis='x', rotation=45)
    
    # Plot 4-7: Scatter plots for each property
    properties = test_results['property'].unique()
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold']
    
    for i, prop in enumerate(properties):  # First 3 properties
        prop_data = test_results[test_results['property'] == prop]
        axes[1,i].scatter(prop_data['target'], prop_data['pred'], 
                         alpha=0.6, color=colors[i])
        
        # Add perfect prediction line
        min_val = min(prop_data['target'].min(), prop_data['pred'].min())
        max_val = max(prop_data['target'].max(), prop_data['pred'].max())
        axes[1,i].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        
        axes[1,i].set_xlabel('True Value')
        axes[1,i].set_ylabel('Predicted Value')
        axes[1,i].set_title(f'{prop} (SRCC: {performance_df[performance_df["property"]==prop]["srcc"].iloc[0]:.3f})')
    
    plt.tight_layout()
    plt.savefig('training_results/analysis/cgcnn_4properties_analysis.png', dpi=300, bbox_inches='tight')
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
    performance_df.to_csv('training_results/analysis/cgcnn_4properties_performance.csv', index=False)
    comparison_df.to_csv('training_results/analysis/cgcnn_4properties_comparison.csv', index=False)
    print("\nResults saved to CSV files")
    
    return performance_df, comparison_df

if __name__ == "__main__":
    performance_results, comparison_results = main()