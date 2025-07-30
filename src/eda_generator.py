#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 00:56:19 2025.

@author: iancj
"""

import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')


class EDAGenerator:
    """
    Generalizable EDA report generator for labeled and unlabeled datasets.

    Produces professional visualizations and comprehensive analysis.
    """

    def __init__(self, data, target_column=None, output_path='eda_report'):
        """
        Initialize EDA generator.

        Args:
        ----
            data: pandas DataFrame
            target_column: str, name of target column (None for unlabeled data)
            output_path: str, path for saving outputs
        """
        self.data = data.copy()
        self.target_column = target_column
        self.output_path = output_path
        self.numeric_columns = []
        self.categorical_columns = []
        self.datetime_columns = []
        self.is_labeled = target_column is not None

        # Set up matplotlib/seaborn styling
        plt.style.use('default')
        sns.set_palette("husl")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10

        self._identify_column_types()

    def _identify_column_types(self):
        """Identify numeric, categorical, and datetime columns."""
        for col in self.data.columns:
            if col == self.target_column:
                continue

            if pd.api.types.is_numeric_dtype(self.data[col]):
                self.numeric_columns.append(col)
            elif pd.api.types.is_datetime64_any_dtype(self.data[col]):
                self.datetime_columns.append(col)
            else:
                self.categorical_columns.append(col)

    def generate_basic_info(self):
        """Generate basic dataset information."""
        print("="*60)
        print("DATASET OVERVIEW")
        print("="*60)

        print(f"Dataset Shape: {self.data.shape}")
        print(f"Dataset Type: {'Labeled' if self.is_labeled else 'Unlabeled'}")
        if self.is_labeled:
            print(f"Target Column: {self.target_column}")

        print(f"\nColumn Types:")
        print(f"  Numeric: {len(self.numeric_columns)}")
        print(f"  Categorical: {len(self.categorical_columns)}")
        print(f"  Datetime: {len(self.datetime_columns)}")

        print(f"\nMemory Usage: {self.data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

        # Missing values summary
        missing_summary = self.data.isnull().sum()
        missing_pct = (missing_summary / len(self.data)) * 100

        if missing_summary.sum() > 0:
            print(f"\nMissing Values:")
            for col in missing_summary[missing_summary > 0].index:
                print(f"  {col}: {missing_summary[col]} ({missing_pct[col]:.1f}%)")
        else:
            print("\nNo missing values found.")

        print("\n" + "="*60)

    def plot_missing_values(self):
        """Visualize missing values pattern."""
        missing_data = self.data.isnull().sum()

        if missing_data.sum() == 0:
            print("No missing values to visualize.")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Missing values bar plot
        missing_cols = missing_data[missing_data > 0].sort_values(ascending=False)
        missing_cols.plot(kind='bar', ax=ax1, color='salmon')
        ax1.set_title('Missing Values by Column', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Number of Missing Values')
        ax1.tick_params(axis='x', rotation=45)

        # Missing values heatmap
        if len(self.data.columns) <= 20:  # Only for manageable number of columns
            sns.heatmap(self.data.isnull(), cbar=True, ax=ax2, cmap='viridis')
            ax2.set_title('Missing Values Heatmap', fontsize=14, fontweight='bold')
        else:
            ax2.text(0.5, 0.5, 'Too many columns\nfor heatmap visualization',
                    ha='center', va='center', transform=ax2.transAxes, fontsize=12)
            ax2.set_title('Missing Values Pattern', fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.show()

    def analyze_numeric_features(self):
        """Analyze numeric features with visualizations."""
        if not self.numeric_columns:
            print("No numeric columns found.")
            return

        print("\n" + "="*60)
        print("NUMERIC FEATURES ANALYSIS")
        print("="*60)

        # Statistical summary
        numeric_stats = self.data[self.numeric_columns].describe()
        print("\nStatistical Summary:")
        print(numeric_stats.round(3))

        # Distribution plots
        n_cols = min(3, len(self.numeric_columns))
        n_rows = (len(self.numeric_columns) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes
        else:
            axes = axes.flatten()

        for i, col in enumerate(self.numeric_columns):
            if i < len(axes):
                sns.histplot(data=self.data, x=col, kde=True, ax=axes[i])
                axes[i].set_title(f'Distribution of {col}', fontweight='bold')
                axes[i].grid(True, alpha=0.3)

        # Hide empty subplots
        for i in range(len(self.numeric_columns), len(axes)):
            axes[i].set_visible(False)

        plt.suptitle('Numeric Features Distributions', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

        # Correlation matrix
        if len(self.numeric_columns) > 1:
            plt.figure(figsize=(10, 8))
            correlation_matrix = self.data[self.numeric_columns].corr()
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

            sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm',
                       center=0, square=True, fmt='.2f')
            plt.title('Numeric Features Correlation Matrix', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.show()

    def analyze_categorical_features(self):
        """Analyze categorical features with visualizations."""
        if not self.categorical_columns:
            print("No categorical columns found.")
            return

        print("\n" + "="*60)
        print("CATEGORICAL FEATURES ANALYSIS")
        print("="*60)

        for col in self.categorical_columns:
            print(f"\n{col}:")
            value_counts = self.data[col].value_counts()
            print(f"  Unique values: {self.data[col].nunique()}")
            print(f"  Most frequent: {value_counts.index[0]} ({value_counts.iloc[0]} times)")

            if self.data[col].nunique() <= 20:  # Only show for manageable number of categories
                print(f"  Value distribution:")
                for val, count in value_counts.head(10).items():
                    pct = (count / len(self.data)) * 100
                    print(f"    {val}: {count} ({pct:.1f}%)")

        # Visualize categorical distributions
        n_cols = min(2, len(self.categorical_columns))
        n_rows = (len(self.categorical_columns) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes
        else:
            axes = axes.flatten()

        for i, col in enumerate(self.categorical_columns):
            if i < len(axes):
                value_counts = self.data[col].value_counts().head(10)

                if len(value_counts) <= 10:
                    sns.countplot(data=self.data, y=col, order=value_counts.index, ax=axes[i])
                    axes[i].set_title(f'Distribution of {col}', fontweight='bold')
                else:
                    axes[i].text(0.5, 0.5, f'{col}\nToo many categories\n({self.data[col].nunique()} unique values)',
                               ha='center', va='center', transform=axes[i].transAxes)
                    axes[i].set_title(f'Distribution of {col}', fontweight='bold')

        # Hide empty subplots
        for i in range(len(self.categorical_columns), len(axes)):
            axes[i].set_visible(False)

        plt.suptitle('Categorical Features Distributions', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

    def analyze_target_relationships(self):
        """Analyze relationships with target variable (for labeled datasets)."""
        if not self.is_labeled:
            return

        print("\n" + "="*60)
        print(f"TARGET VARIABLE ANALYSIS: {self.target_column}")
        print("="*60)

        target_data = self.data[self.target_column]

        # Target distribution
        print(f"\nTarget Variable Distribution:")
        if pd.api.types.is_numeric_dtype(target_data):
            print(f"  Type: Numeric (Regression)")
            print(f"  Mean: {target_data.mean():.3f}")
            print(f"  Std: {target_data.std():.3f}")
            print(f"  Min: {target_data.min():.3f}")
            print(f"  Max: {target_data.max():.3f}")
        else:
            print(f"  Type: Categorical (Classification)")
            print(f"  Classes: {target_data.nunique()}")
            value_counts = target_data.value_counts()
            for val, count in value_counts.items():
                pct = (count / len(target_data)) * 100
                print(f"    {val}: {count} ({pct:.1f}%)")

        # Target distribution plot
        plt.figure(figsize=(12, 5))

        if pd.api.types.is_numeric_dtype(target_data):
            plt.subplot(1, 2, 1)
            sns.histplot(target_data, kde=True)
            plt.title(f'Distribution of {self.target_column}', fontweight='bold')
            plt.grid(True, alpha=0.3)

            plt.subplot(1, 2, 2)
            stats.probplot(target_data.dropna(), dist="norm", plot=plt)
            plt.title(f'Q-Q Plot of {self.target_column}', fontweight='bold')
            plt.grid(True, alpha=0.3)
        else:
            plt.subplot(1, 2, 1)
            sns.countplot(data=self.data, x=self.target_column)
            plt.title(f'Distribution of {self.target_column}', fontweight='bold')
            plt.xticks(rotation=45)

            plt.subplot(1, 2, 2)
            target_data.value_counts().plot(kind='pie', autopct='%1.1f%%')
            plt.title(f'Proportion of {self.target_column}', fontweight='bold')
            plt.ylabel('')

        plt.tight_layout()
        plt.show()

        # Feature-target relationships
        self._plot_feature_target_relationships()

    def _plot_feature_target_relationships(self):
        """Plot relationships between features and target."""
        target_data = self.data[self.target_column]
        is_target_numeric = pd.api.types.is_numeric_dtype(target_data)

        # Numeric features vs target
        if self.numeric_columns:
            n_cols = min(3, len(self.numeric_columns))
            n_rows = (len(self.numeric_columns) + n_cols - 1) // n_cols

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
            if n_rows == 1 and n_cols == 1:
                axes = [axes]
            elif n_rows == 1:
                axes = axes
            else:
                axes = axes.flatten()

            for i, col in enumerate(self.numeric_columns):
                if i < len(axes):
                    if is_target_numeric:
                        sns.scatterplot(data=self.data, x=col, y=self.target_column,
                                      alpha=0.6, ax=axes[i])
                        axes[i].set_title(f'{col} vs {self.target_column}', fontweight='bold')
                    else:
                        sns.boxplot(data=self.data, x=self.target_column, y=col, ax=axes[i])
                        axes[i].set_title(f'{col} by {self.target_column}', fontweight='bold')
                        axes[i].tick_params(axis='x', rotation=45)

                    axes[i].grid(True, alpha=0.3)

            # Hide empty subplots
            for i in range(len(self.numeric_columns), len(axes)):
                axes[i].set_visible(False)

            plt.suptitle('Numeric Features vs Target', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.show()

        # Categorical features vs target
        if self.categorical_columns:
            for col in self.categorical_columns[:6]:  # Limit to first 6 for readability
                plt.figure(figsize=(12, 5))

                if is_target_numeric:
                    plt.subplot(1, 2, 1)
                    sns.boxplot(data=self.data, x=col, y=self.target_column)
                    plt.title(f'{self.target_column} by {col}', fontweight='bold')
                    plt.xticks(rotation=45)

                    plt.subplot(1, 2, 2)
                    sns.violinplot(data=self.data, x=col, y=self.target_column)
                    plt.title(f'{self.target_column} Distribution by {col}', fontweight='bold')
                    plt.xticks(rotation=45)
                else:
                    plt.subplot(1, 2, 1)
                    pd.crosstab(self.data[col], self.data[self.target_column]).plot(kind='bar', ax=plt.gca())
                    plt.title(f'{col} vs {self.target_column}', fontweight='bold')
                    plt.xticks(rotation=45)
                    plt.legend(title=self.target_column)

                    plt.subplot(1, 2, 2)
                    pd.crosstab(self.data[col], self.data[self.target_column], normalize='index').plot(kind='bar', stacked=True, ax=plt.gca())
                    plt.title(f'{col} vs {self.target_column} (Normalized)', fontweight='bold')
                    plt.xticks(rotation=45)
                    plt.legend(title=self.target_column)

                plt.tight_layout()
                plt.show()

    def detect_outliers(self):
        """Detect and visualize outliers in numeric features."""
        if not self.numeric_columns:
            return

        print("\n" + "="*60)
        print("OUTLIER ANALYSIS")
        print("="*60)

        # Box plots for outlier detection
        n_cols = min(3, len(self.numeric_columns))
        n_rows = (len(self.numeric_columns) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes
        else:
            axes = axes.flatten()

        outlier_summary = {}

        for i, col in enumerate(self.numeric_columns):
            if i < len(axes):
                # Box plot
                sns.boxplot(data=self.data, y=col, ax=axes[i])
                axes[i].set_title(f'Outliers in {col}', fontweight='bold')
                axes[i].grid(True, alpha=0.3)

                # Calculate outliers using IQR method
                Q1 = self.data[col].quantile(0.25)
                Q3 = self.data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                outliers = self.data[(self.data[col] < lower_bound) | (self.data[col] > upper_bound)]
                outlier_summary[col] = len(outliers)

        # Hide empty subplots
        for i in range(len(self.numeric_columns), len(axes)):
            axes[i].set_visible(False)

        plt.suptitle('Outlier Detection (Box Plots)', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

        # Print outlier summary
        print("\nOutlier Summary (IQR method):")
        for col, count in outlier_summary.items():
            pct = (count / len(self.data)) * 100
            print(f"  {col}: {count} outliers ({pct:.1f}%)")

    def generate_summary_report(self):
        """Generate a comprehensive summary report."""
        print("\n" + "="*60)
        print("SUMMARY REPORT")
        print("="*60)

        print(f"\nDataset: {self.data.shape[0]} rows × {self.data.shape[1]} columns")
        print(f"Type: {'Supervised Learning' if self.is_labeled else 'Unsupervised Learning'}")

        if self.is_labeled:
            target_type = "Regression" if pd.api.types.is_numeric_dtype(self.data[self.target_column]) else "Classification"
            print(f"Task: {target_type}")

        print(f"\nFeature Types:")
        print(f"  • Numeric: {len(self.numeric_columns)}")
        print(f"  • Categorical: {len(self.categorical_columns)}")
        print(f"  • Datetime: {len(self.datetime_columns)}")

        # Data quality assessment
        missing_pct = (self.data.isnull().sum().sum() / (len(self.data) * len(self.data.columns))) * 100
        print(f"\nData Quality:")
        print(f"  • Overall missing values: {missing_pct:.1f}%")

        # Memory and computational considerations
        memory_mb = self.data.memory_usage(deep=True).sum() / 1024**2
        print(f"  • Memory usage: {memory_mb:.1f} MB")

        if memory_mb > 100:
            print("  ⚠️  Large dataset - consider sampling for initial analysis")

        high_cardinality_cols = [col for col in self.categorical_columns
                               if self.data[col].nunique() > 50]
        if high_cardinality_cols:
            print(f"  ⚠️  High cardinality categorical features: {high_cardinality_cols}")

        print(f"\nNext Steps Recommendations:")
        print(f"  1. Address missing values using appropriate imputation strategies")
        print(f"  2. Consider feature engineering for categorical variables")
        print(f"  3. Scale/normalize numeric features for model training")

        if self.is_labeled:
            print(f"  4. Split data into train/validation/test sets")
            print(f"  5. Consider cross-validation strategy")
        else:
            print(f"  4. Consider dimensionality reduction techniques")
            print(f"  5. Explore clustering approaches")

        print("="*60)

    def run_complete_eda(self):
        """Run the complete EDA pipeline."""
        print("Starting Comprehensive EDA Analysis...")
        print("This may take a few moments for large datasets.\n")

        # Basic information
        self.generate_basic_info()

        # Missing values analysis
        self.plot_missing_values()

        # Feature analysis
        self.analyze_numeric_features()
        self.analyze_categorical_features()

        # Target analysis (if labeled)
        if self.is_labeled:
            self.analyze_target_relationships()

        # Outlier detection
        self.detect_outliers()

        # Summary report
        self.generate_summary_report()

        print("\n✅ EDA Analysis Complete!")
        print(f"Report generated for {'labeled' if self.is_labeled else 'unlabeled'} dataset")


# Usage Examples:
"""
# For labeled dataset (supervised learning)
df_labeled = pd.read_csv('your_labeled_data.csv')
eda_labeled = EDAGenerator(df_labeled, target_column='target')
eda_labeled.run_complete_eda()

# For unlabeled dataset (unsupervised learning)
df_unlabeled = pd.read_csv('your_unlabeled_data.csv')
eda_unlabeled = EDAGenerator(df_unlabeled)
eda_unlabeled.run_complete_eda()

# Run specific analyses
eda_labeled.analyze_numeric_features()
eda_labeled.detect_outliers()
eda_labeled.analyze_target_relationships()
"""
