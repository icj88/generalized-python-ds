#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 23:58:42 2025

@author: iancj

Example script demonstrating how to use the EDA framework
for both labeled and unlabeled datasets.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression
from eda_generator import EDAGenerator
from eda_utilities import EDAUtilities


# %%
def create_sample_datasets():
    """Create sample datasets for demonstration."""

    # Create a classification dataset
    X_class, y_class = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=7,
        n_redundant=2,
        n_clusters_per_class=1,
        random_state=42
    )

    # Create feature names
    feature_names = [f'feature_{i}' for i in range(X_class.shape[1])]

    # Create DataFrame for classification
    df_classification = pd.DataFrame(X_class, columns=feature_names)
    df_classification['target'] = y_class

    # Add some categorical features
    df_classification['category_A'] = np.random.choice(['A', 'B', 'C'], size=1000)
    df_classification['category_B'] = np.random.choice(['X', 'Y'], size=1000, p=[0.7, 0.3])

    # Add some missing values
    missing_indices = np.random.choice(1000, 50, replace=False)
    df_classification.loc[missing_indices, 'feature_3'] = np.nan

    # Create a regression dataset
    X_reg, y_reg = make_regression(
        n_samples=1000,
        n_features=8,
        noise=0.1,
        random_state=42
    )

    # Create DataFrame for regression
    reg_feature_names = [f'numeric_{i}' for i in range(X_reg.shape[1])]
    df_regression = pd.DataFrame(X_reg, columns=reg_feature_names)
    df_regression['price'] = y_reg

    # Add categorical features
    df_regression['region'] = np.random.choice(['North', 'South', 'East', 'West'], size=1000)
    df_regression['size'] = np.random.choice(['Small', 'Medium', 'Large'], size=1000, p=[0.3, 0.5, 0.2])

    # Create an unlabeled dataset (just features)
    df_unlabeled = df_classification.drop('target', axis=1)

    return df_classification, df_regression, df_unlabeled

def run_classification_eda(df):
    """Run complete EDA on classification dataset."""
    print("\n" + "="*80)
    print("CLASSIFICATION DATASET EDA")
    print("="*80)

    # Initialize EDA generator
    eda = EDAGenerator(df, target_column='target')

    # Run complete EDA
    eda.run_complete_eda()

    # Additional utility analyses
    print("\n" + "-"*60)
    print("ADDITIONAL ANALYSES")
    print("-"*60)

    # Feature importance
    importance_df = EDAUtilities.feature_importance_analysis(df, 'target')
    print(f"\nTop 5 Most Important Features:")
    print(importance_df.head().round(3))

    # Correlation analysis
    high_corr = EDAUtilities.correlation_analysis(df)

    # Data quality report
    quality_df = EDAUtilities.data_quality_report(df)

    return eda, importance_df, quality_df

def run_regression_eda(df):
    """Run complete EDA on regression dataset."""
    print("\n" + "="*80)
    print("REGRESSION DATASET EDA")
    print("="*80)

    # Initialize EDA generator
    eda = EDAGenerator(df, target_column='price')

    # Run complete EDA
    eda.run_complete_eda()

    # Additional analyses
    print("\n" + "-"*60)
    print("ADDITIONAL ANALYSES")
    print("-"*60)

    # Distribution analysis
    dist_df = EDAUtilities.distribution_analysis(df, test_normality=True)

    # Outlier analysis
    outlier_df = EDAUtilities.outlier_treatment_analysis(df)

    # Feature engineering suggestions
    suggestions = EDAUtilities.feature_engineering_suggestions(df, 'price')

    return eda, dist_df, outlier_df

def run_unsupervised_eda(df):
    """Run EDA on unlabeled dataset."""
    print("\n" + "="*80)
    print("UNSUPERVISED DATASET EDA")
    print("="*80)

    # Initialize EDA generator (no target column)
    eda = EDAGenerator(df)

    # Run complete EDA
    eda.run_complete_eda()

    # Additional analyses for unsupervised data
    print("\n" + "-"*60)
    print("ADDITIONAL ANALYSES")
    print("-"*60)

    # Categorical encoding analysis
    encoding_df = EDAUtilities.categorical_encoding_analysis(df)

    # Correlation analysis to find feature relationships
    high_corr = EDAUtilities.correlation_analysis(df, threshold=0.7)

    return eda, encoding_df

def comprehensive_eda_example():
    """Run comprehensive EDA example on all dataset types."""

    print("Creating sample datasets...")
    df_class, df_reg, df_unlab = create_sample_datasets()

    print(f"Classification dataset shape: {df_class.shape}")
    print(f"Regression dataset shape: {df_reg.shape}")
    print(f"Unlabeled dataset shape: {df_unlab.shape}")

    # Run classification EDA
    class_eda, class_importance, class_quality = run_classification_eda(df_class)

    # Run regression EDA
    reg_eda, reg_dist, reg_outliers = run_regression_eda(df_reg)

    # Run unsupervised EDA
    unlab_eda, unlab_encoding = run_unsupervised_eda(df_unlab)

    print("\n" + "="*80)
    print("EDA ANALYSIS COMPLETE")
    print("="*80)
    print("All analyses have been completed. The framework provides:")
    print("1. Comprehensive visualizations for data understanding")
    print("2. Statistical summaries and feature analysis")
    print("3. Data quality assessment and recommendations")
    print("4. Feature engineering suggestions")
    print("5. Specialized analyses for different ML tasks")

def analyze_custom_dataset(file_path, target_column=None):
    """
    Analyze a custom dataset using the EDA framework.

    Args:
        file_path: str, path to CSV file
        target_column: str, name of target column (None for unsupervised)
    """
    try:
        # Load data
        df = pd.read_csv(file_path)
        print(f"Loaded dataset: {df.shape}")

        # Initialize EDA
        eda = EDAGenerator(df, target_column=target_column)

        # Run complete analysis
        eda.run_complete_eda()

        # Additional utility analyses
        if target_column:
            importance_df = EDAUtilities.feature_importance_analysis(df, target_column)
            suggestions = EDAUtilities.feature_engineering_suggestions(df, target_column)

        quality_df = EDAUtilities.data_quality_report(df)

        return eda, quality_df

    except Exception as e:
        print(f"Error analyzing dataset: {str(e)}")
        return None, None

if __name__ == "__main__":
    # Run the comprehensive example
    comprehensive_eda_example()

    # Example of analyzing a custom dataset
    # eda, quality = analyze_custom_dataset('your_dataset.csv', target_column='your_target')

    print("\n" + "="*80)
    print("USAGE INSTRUCTIONS")
    print("="*80)
    print("To use this framework with your own data:")
    print()
    print("1. For supervised learning (with target variable):")
    print("   eda = EDAGenerator(your_df, target_column='target_name')")
    print("   eda.run_complete_eda()")
    print()
    print("2. For unsupervised learning (no target variable):")
    print("   eda = EDAGenerator(your_df)")
    print("   eda.run_complete_eda()")
    print()
    print("3. For specific analyses:")
    print("   EDAUtilities.feature_importance_analysis(df, 'target')")
    print("   EDAUtilities.correlation_analysis(df)")
    print("   EDAUtilities.data_quality_report(df)")
    print()
    print("4. For custom datasets from file:")
    print("   analyze_custom_dataset('path/to/file.csv', 'target_column')")
    print("="*80)
