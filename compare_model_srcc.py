#!/usr/bin/env python3
"""
Script to compare SRCC (Spearman Rank Correlation Coefficient) across 4 different models/representations.

This script:
1. Calculates SRCC from raw prediction files for CV models (CGCNN, Transformer, PointNet)
2. Calculates mean and standard deviation of SRCC across 5 folds
3. Calculates SRCC for LLM model (single test result)
4. Generates a bar plot comparing all 4 models across 6 properties
5. Saves results to CSV and plot to image

Models:
- CGCNN (5-fold CV)
- Transformer (5-fold CV) 
- PointNet (5-fold CV)
- LLM (single test)

Properties (in order): Di, Dif, Df, CH4_HP, CO2_LP, logKH_CO2
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration
PROPERTIES = ['Di', 'Dif', 'Df', 'CH4_HP', 'CO2_LP', 'logKH_CO2']
MODELS = ['CGCNN', 'Transformer', 'PointNet', 'LLM']
N_FOLDS = 5

# Base paths
BASE_PATH = Path('training_results/finetuning')
CV_PATHS = {
    'CGCNN': BASE_PATH / 'CGCNN_CV',
    'Transformer': BASE_PATH / 'Transformer_CV', 
    'PointNet': BASE_PATH / 'PointNet_CV'
}

# Placeholder path for LLM results - update this when results are available
LLM_BASE_PATH = BASE_PATH / 'LLM_MOF'  # Update this path as needed

# Color scheme for models
MODEL_COLORS = {
    'CGCNN': '#2E86C1',      # Blue
    'Transformer': '#E74C3C', # Red  
    'PointNet': '#28B463',    # Green
    'LLM': '#F39C12'          # Orange
}

def calculate_srcc_from_predictions(y_true, y_pred):
    """Calculate Spearman Rank Correlation Coefficient."""
    try:
        srcc, _ = spearmanr(y_true, y_pred)
        return srcc
    except:
        return np.nan

def load_cv_results(model_name, property_name):
    """Load cross-validation results for a given model and property."""
    results = []
    base_path = CV_PATHS[model_name]
    
    for fold in range(N_FOLDS):
        fold_dir = base_path / f'{model_name}_fold_{fold}_{property_name}'
        result_file = fold_dir / f'test_results_{property_name}.csv'
        
        if result_file.exists():
            try:
                df = pd.read_csv(result_file)
                if 'target' in df.columns and 'pred' in df.columns:
                    srcc, p_val = calculate_srcc_from_predictions(df['target'], df['pred'])
                    results.append({
                        'fold': fold,
                        'srcc': srcc,
                        'n_samples': len(df)
                    })
                else:
                    print(f"Warning: Missing target/pred columns in {result_file}")
            except Exception as e:
                print(f"Error loading {result_file}: {e}")
        else:
            print(f"Warning: File not found {result_file}")
    
    return results

def load_llm_results(property_name):
    """Load LLM results for a given property."""
    # This is a placeholder - update the path structure when LLM results are available
    llm_file = LLM_BASE_PATH / f'gpt-4o-mini-2024-07-18_CoRE2019_1_{property_name}' / f'test_results_{property_name}.csv'
    
    if llm_file.exists():
        try:
            df = pd.read_csv(llm_file)
            if 'target' in df.columns and 'pred' in df.columns:
                srcc, _ = calculate_srcc_from_predictions(df['target'], df['pred'])
                return {
                    'srcc': srcc,
                    'n_samples': len(df)
                }
            else:
                print(f"Warning: Missing target/pred columns in {llm_file}")
        except Exception as e:
            print(f"Error loading {llm_file}: {e}")
    else:
        print(f"Warning: LLM file not found {llm_file}")

    return {'srcc': np.nan, 'n_samples': 0}

def calculate_cv_statistics(cv_results):
    """Calculate mean and standard deviation for CV results."""
    if not cv_results:
        return {'mean': np.nan, 'std': np.nan, 'n_folds': 0}
    
    srcc_values = [r['srcc'] for r in cv_results if not np.isnan(r['srcc'])]
    
    if not srcc_values:
        return {'mean': np.nan, 'std': np.nan, 'n_folds': 0}
    
    return {
        'mean': np.mean(srcc_values),
        'std': np.std(srcc_values, ddof=1) if len(srcc_values) > 1 else 0.0,
        'n_folds': len(srcc_values)
    }

def collect_all_results():
    """Collect SRCC results for all models and properties."""
    all_results = []
    
    print("Collecting results for all models and properties...")
    
    for property_name in PROPERTIES:
        print(f"\nProcessing property: {property_name}")
        
        # Process CV models
        for model_name in ['CGCNN', 'Transformer', 'PointNet']:
            print(f"  Loading {model_name} CV results...")
            cv_results = load_cv_results(model_name, property_name)
            stats = calculate_cv_statistics(cv_results)
            
            all_results.append({
                'Property': property_name,
                'Model': model_name,
                'SRCC_mean': stats['mean'],
                'SRCC_std': stats['std'],
                'n_folds': stats['n_folds'],
                'model_type': 'CV'
            })
            
            print(f"    {model_name}: SRCC = {stats['mean']:.3f} ± {stats['std']:.3f} (n={stats['n_folds']})")
        
        # Process LLM model
        print(f"  Loading LLM results...")
        llm_result = load_llm_results(property_name)
        
        all_results.append({
            'Property': property_name,
            'Model': 'LLM',
            'SRCC_mean': llm_result['srcc'],
            'SRCC_std': 0.0,  # No standard deviation for single test
            'n_folds': 1 if not np.isnan(llm_result['srcc']) else 0,
            'model_type': 'single'
        })
        
        print(f"    LLM: SRCC = {llm_result['srcc']:.3f}")
    
    return pd.DataFrame(all_results)

def create_comparison_plot(results_df, save_path='training_results/cross_model_analysis/model_srcc_comparison.png'):
    """Create a bar plot comparing SRCC across models and properties."""
    # Set up the plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Prepare data for plotting
    properties = PROPERTIES
    n_properties = len(properties)
    n_models = len(MODELS)
    
    # Set up bar positions
    bar_width = 0.2
    x_positions = np.arange(n_properties)
    
    # Create bars for each model
    for i, model in enumerate(MODELS):
        model_data = results_df[results_df['Model'] == model]
        
        # Ensure properties are in correct order
        srcc_values = []
        std_values = []
        
        for prop in properties:
            prop_data = model_data[model_data['Property'] == prop]
            if not prop_data.empty:
                srcc_values.append(prop_data['SRCC_mean'].iloc[0])
                std_values.append(prop_data['SRCC_std'].iloc[0])
            else:
                srcc_values.append(np.nan)
                std_values.append(0.0)
        
        # Plot bars with error bars only for CV models (not LLM)
        if model == 'LLM':
            # No error bars for LLM (single test result)
            bars = ax.bar(x_positions + i * bar_width, 
                         srcc_values,
                         bar_width,
                         label=model,
                         color=MODEL_COLORS[model],
                         alpha=0.8)
        else:
            # Error bars for CV models
            bars = ax.bar(x_positions + i * bar_width, 
                         srcc_values,
                         bar_width,
                         yerr=std_values,
                         label=model,
                         color=MODEL_COLORS[model],
                         alpha=0.8,
                         capsize=5)
    
    # Customize the plot
    ax.set_xlabel('Properties', fontsize=12, fontweight='bold')
    ax.set_ylabel('Spearman Rank Correlation Coefficient (SRCC)', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Comparison Across MOF Properties', fontsize=14, fontweight='bold')
    
    # Set x-axis
    ax.set_xticks(x_positions + bar_width * (n_models - 1) / 2)
    ax.set_xticklabels(properties, fontsize=11)
    
    # Set y-axis
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)
    
    # Add legend
    ax.legend(fontsize=11, loc='upper right')
    
    # Adjust layout to prevent cutoff
    plt.tight_layout()
    
    # Add value labels on bars (optional)
    for i, model in enumerate(MODELS):
        model_data = results_df[results_df['Model'] == model]
        for j, prop in enumerate(properties):
            prop_data = model_data[model_data['Property'] == prop]
            if not prop_data.empty and not np.isnan(prop_data['SRCC_mean'].iloc[0]):
                value = prop_data['SRCC_mean'].iloc[0]
                ax.text(j + i * bar_width, value + 0.02, f'{value:.3f}', 
                       ha='center', va='bottom', fontsize=8, rotation=0)
    
    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")
    
    return fig, ax

def save_results_csv(results_df, save_path='model_srcc_results.csv'):
    """Save results to CSV file."""
    # Pivot the data for better readability
    pivot_df = results_df.pivot_table(
        index='Property', 
        columns='Model', 
        values=['SRCC_mean', 'SRCC_std'], 
        aggfunc='first'
    )
    
    # Flatten column names
    pivot_df.columns = [f'{col[1]}_{col[0]}' for col in pivot_df.columns]
    
    # Reorder columns
    ordered_cols = []
    for model in MODELS:
        ordered_cols.extend([f'{model}_SRCC_mean', f'{model}_SRCC_std'])
    
    pivot_df = pivot_df[ordered_cols]
    
    # Reorder rows according to PROPERTIES order
    pivot_df = pivot_df.reindex(PROPERTIES)
    
    # Save to CSV
    pivot_df.to_csv(save_path)
    print(f"Results saved to: {save_path}")
    
    # Also save the raw results
    raw_save_path = save_path.replace('.csv', '_raw.csv')
    results_df.to_csv(raw_save_path, index=False)
    print(f"Raw results saved to: {raw_save_path}")
    
    return pivot_df

def print_summary(results_df):
    """Print a summary of the results."""
    print("\n" + "="*60)
    print("SUMMARY OF SRCC RESULTS")
    print("="*60)
    
    for prop in PROPERTIES:
        print(f"\n{prop}:")
        prop_data = results_df[results_df['Property'] == prop]
        
        for model in MODELS:
            model_data = prop_data[prop_data['Model'] == model]
            if not model_data.empty:
                mean_val = model_data['SRCC_mean'].iloc[0]
                std_val = model_data['SRCC_std'].iloc[0]
                n_folds = model_data['n_folds'].iloc[0]
                
                if model == 'LLM':
                    print(f"  {model:12}: {mean_val:.3f}")
                else:
                    print(f"  {model:12}: {mean_val:.3f} ± {std_val:.3f} (n={n_folds})")
            else:
                print(f"  {model:12}: No data")
    
    print("\n" + "="*60)

def main():
    """Main function to run the analysis."""
    print("Starting SRCC comparison analysis...")
    print(f"Properties: {PROPERTIES}")
    print(f"Models: {MODELS}")
    
    # Collect all results
    results_df = collect_all_results()
    
    # Print summary
    print_summary(results_df)
    
    # Create comparison plot
    fig, ax = create_comparison_plot(results_df, 'model_srcc_comparison.png')
    
    # Save results to CSV
    pivot_df = save_results_csv(results_df, 'model_srcc_results.csv')
    
    # Display the pivot table
    print("\nPivot Table of Results:")
    print(pivot_df.round(3))
    
    # Show the plot
    plt.show()
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
