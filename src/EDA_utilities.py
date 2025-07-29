#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 20:54:54 2025.

@author: iancj
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')


class EDAUtilities:
    """
    Additional utility functions for specialized EDA tasks.

    Complements the main EDAGenerator class.
    """

    @staticmethod
    def feature_importance_analysis(data: pd.DataFrame,
                                    target_column: str,
                                    feature_columns: list = None,
                                    task_type: str = 'auto'):
        """
        Analyze feature importance using mutual information.

        Args:
        ----
            data: pandas DataFrame
            target_column: str, target variable name
            feature_columns: list, features to analyze (None for all)
            task_type: str, 'regression', 'classification', or 'auto'
        """
        if feature_columns is None:
            feature_columns = [col for col in data.columns if col != target_column]

        # Prepare data
        X = data[feature_columns].copy()
        y = data[target_column].copy()

        # Handle categorical variables
        categorical_cols = X.select_dtypes(include=['object']).columns
        le_dict = {}

        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            le_dict[col] = le
        # Handle missing values
        X = X.fillna(X.median() if X.select_dtypes(include=[np.number]).shape[1] > 0 else X.mode().iloc[0])
        y = y.fillna(y.median() if pd.api.types.is_numeric_dtype(y) else y.mode().iloc[0])

        # Determine task type
        if task_type == 'auto':
            task_type = 'regression' if pd.api.types.is_numeric_dtype(y) else 'classification'

        # Calculate mutual information
        if task_type == 'regression':
            mi_scores = mutual_info_regression(X, y, random_state=42)
        else:
            if not pd.api.types.is_numeric_dtype(y):
                le_target = LabelEncoder()
                y = le_target.fit_transform(y.astype(str))
            mi_scores = mutual_info_classif(X, y, random_state=42)

        # Create results DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_columns,
            'importance': mi_scores
        }).sort_values('importance', ascending=False)

        # Visualization
        plt.figure(figsize=(12, 8))
        sns.barplot(data=importance_df.head(20), x='importance', y='feature', palette='viridis')
        plt.title(f'Feature Importance - {task_type.capitalize()} Task', fontsize=14, fontweight='bold')
        plt.xlabel('Mutual Information Score')
        plt.tight_layout()
        plt.show()

        return importance_df

    @staticmethod
    def correlation_analysis(data, threshold=0.8, method='pearson'):
        """
        Advanced correlation analysis with high correlation detection.

        Args:
        ----
            data: pandas DataFrame
            threshold: float, correlation threshold for flagging
            method: str, correlation method ('pearson', 'spearman', 'kendall')
        """
        numeric_data = data.select_dtypes(include=[np.number])

        if numeric_data.empty:
            print("No numeric columns found for correlation analysis.")
            return None

        # Calculate correlation matrix
        corr_matrix = numeric_data.corr(method=method)

        # Find highly correlated pairs
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) >= threshold:
                    high_corr_pairs.append({
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.columns[j],
                        'correlation': corr_matrix.iloc[i, j]
                    })

        # Visualization
        plt.figure(figsize=(12, 10))

        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

        # Generate heatmap
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.2f', cbar_kws={"shrink": .8})
        plt.title(f'Correlation Matrix ({method.capitalize()})', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

        # Print high correlations
        if high_corr_pairs:
            print(f"\nHighly Correlated Feature Pairs (|r| >= {threshold}):")
            for pair in sorted(high_corr_pairs, key=lambda x: abs(x['correlation']), reverse=True):
                print(f"  {pair['feature1']} ↔ {pair['feature2']}: {pair['correlation']:.3f}")
        else:
            print(f"\nNo highly correlated pairs found (threshold: {threshold})")

        return pd.DataFrame(high_corr_pairs) if high_corr_pairs else None

    @staticmethod
    def distribution_analysis(data, columns=None, test_normality=True):
        """
        Detailed distribution analysis with normality testing.

        Args:
            data: pandas DataFrame
            columns: list, columns to analyze (None for all numeric)
            test_normality: bool, whether to perform normality tests
        """
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()

        results = {}

        for col in columns:
            series = data[col].dropna()

            if series.empty:
                continue

            # Basic statistics
            stats_dict = {
                'count': len(series),
                'mean': series.mean(),
                'median': series.median(),
                'std': series.std(),
                'skewness': stats.skew(series),
                'kurtosis': stats.kurtosis(series),
                'min': series.min(),
                'max': series.max(),
                'range': series.max() - series.min()
            }

            # Normality tests
            if test_normality and len(series) >= 8:  # Minimum sample size for tests
                try:
                    shapiro_stat, shapiro_p = stats.shapiro(series.sample(min(5000, len(series))))
                    stats_dict['shapiro_p_value'] = shapiro_p
                    stats_dict['is_normal_shapiro'] = shapiro_p > 0.05

                    ks_stat, ks_p = stats.kstest(series, 'norm', args=(series.mean(), series.std()))
                    stats_dict['ks_p_value'] = ks_p
                    stats_dict['is_normal_ks'] = ks_p > 0.05
                except:
                    stats_dict['shapiro_p_value'] = None
                    stats_dict['is_normal_shapiro'] = None
                    stats_dict['ks_p_value'] = None
                    stats_dict['is_normal_ks'] = None

            results[col] = stats_dict

        # Create summary DataFrame
        results_df = pd.DataFrame(results).T

        # Visualization
        n_cols = min(3, len(columns))
        n_rows = (len(columns) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows * 2, n_cols, figsize=(5*n_cols, 8*n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes[0]], [axes[1]]])
        elif n_rows == 1:
            axes = axes.reshape(2, n_cols)

        for i, col in enumerate(columns):
            if i < n_cols * n_rows:
                row = i // n_cols
                col_idx = i % n_cols

                series = data[col].dropna()

                # Distribution plot
                sns.histplot(series, kde=True, ax=axes[row*2, col_idx])
                axes[row*2, col_idx].set_title(f'Distribution of {col}', fontweight='bold')
                axes[row*2, col_idx].grid(True, alpha=0.3)

                # Q-Q plot
                stats.probplot(series, dist="norm", plot=axes[row*2+1, col_idx])
                axes[row*2+1, col_idx].set_title(f'Q-Q Plot of {col}', fontweight='bold')
                axes[row*2+1, col_idx].grid(True, alpha=0.3)

        plt.suptitle('Distribution Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

        return results_df

    @staticmethod
    def categorical_encoding_analysis(data, categorical_columns=None, target_column=None):
        """
        Analyze categorical variables and suggest encoding strategies.

        Args:
            data: pandas DataFrame
            categorical_columns: list, categorical columns to analyze
            target_column: str, target variable for supervised analysis
        """
        if categorical_columns is None:
            categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
            if target_column and target_column in categorical_columns:
                categorical_columns.remove(target_column)

        encoding_recommendations = {}

        print("CATEGORICAL ENCODING ANALYSIS")
        print("="*50)

        for col in categorical_columns:
            analysis = {}
            series = data[col]

            # Basic stats
            analysis['unique_count'] = series.nunique()
            analysis['null_count'] = series.isnull().sum()
            analysis['null_percentage'] = (analysis['null_count'] / len(series)) * 100
            analysis['most_frequent'] = series.mode().iloc[0] if not series.mode().empty else None
            analysis['least_frequent_count'] = series.value_counts().min()

            # Encoding recommendations
            if analysis['unique_count'] <= 2:
                analysis['recommended_encoding'] = 'Binary/Label Encoding'
            elif analysis['unique_count'] <= 10:
                analysis['recommended_encoding'] = 'One-Hot Encoding'
            elif analysis['unique_count'] <= 50:
                analysis['recommended_encoding'] = 'Target/Mean Encoding (if supervised)'
            else:
                analysis['recommended_encoding'] = 'Feature Hashing/Embedding'

            # Target relationship (if available)
            if target_column and target_column in data.columns:
                try:
                    if pd.api.types.is_numeric_dtype(data[target_column]):
                        # Regression case
                        grouped = data.groupby(col)[target_column].agg(['mean', 'std', 'count'])
                        analysis['target_variance'] = grouped['mean'].std()
                    else:
                        # Classification case
                        contingency = pd.crosstab(data[col], data[target_column])
                        chi2, p_value, _, _ = stats.chi2_contingency(contingency)
                        analysis['chi2_p_value'] = p_value
                        analysis['significant_association'] = p_value < 0.05
                except:
                    pass

            encoding_recommendations[col] = analysis

            # Print summary
            print(f"\n{col}:")
            print(f"  Unique values: {analysis['unique_count']}")
            print(f"  Missing values: {analysis['null_count']} ({analysis['null_percentage']:.1f}%)")
            print(f"  Recommended encoding: {analysis['recommended_encoding']}")

            if 'significant_association' in analysis:
                association = "Yes" if analysis['significant_association'] else "No"
                print(f"  Significant target association: {association}")

        return pd.DataFrame(encoding_recommendations).T

    @staticmethod
    def outlier_treatment_analysis(data, numeric_columns=None, methods=['iqr', 'zscore', 'isolation']):
        """
        Comprehensive outlier detection and treatment analysis.

        Args:
            data: pandas DataFrame
            numeric_columns: list, columns to analyze
            methods: list, outlier detection methods to use
        """
        if numeric_columns is None:
            numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()

        outlier_summary = {}

        for col in numeric_columns:
            series = data[col].dropna()
            outliers_dict = {}

            # IQR method
            if 'iqr' in methods:
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                iqr_outliers = series[(series < lower_bound) | (series > upper_bound)]
                outliers_dict['iqr_count'] = len(iqr_outliers)
                outliers_dict['iqr_percentage'] = (len(iqr_outliers) / len(series)) * 100

            # Z-score method
            if 'zscore' in methods:
                z_scores = np.abs(stats.zscore(series))
                zscore_outliers = series[z_scores > 3]
                outliers_dict['zscore_count'] = len(zscore_outliers)
                outliers_dict['zscore_percentage'] = (len(zscore_outliers) / len(series)) * 100

            # Isolation Forest (if sklearn available)
            if 'isolation' in methods:
                try:
                    from sklearn.ensemble import IsolationForest
                    iso_forest = IsolationForest(contamination=0.1, random_state=42)
                    outlier_pred = iso_forest.fit_predict(series.values.reshape(-1, 1))
                    isolation_outliers = series[outlier_pred == -1]
                    outliers_dict['isolation_count'] = len(isolation_outliers)
                    outliers_dict['isolation_percentage'] = (len(isolation_outliers) / len(series)) * 100
                except ImportError:
                    outliers_dict['isolation_count'] = None
                    outliers_dict['isolation_percentage'] = None

            outlier_summary[col] = outliers_dict

        # Visualization
        fig, axes = plt.subplots(2, len(numeric_columns), figsize=(5*len(numeric_columns), 10))
        if len(numeric_columns) == 1:
            axes = axes.reshape(-1, 1)

        for i, col in enumerate(numeric_columns):
            series = data[col].dropna()

            # Box plot
            sns.boxplot(y=series, ax=axes[0, i])
            axes[0, i].set_title(f'Box Plot: {col}', fontweight='bold')
            axes[0, i].grid(True, alpha=0.3)

            # Histogram with outlier highlighting
            axes[1, i].hist(series, bins=50, alpha=0.7, color='skyblue', edgecolor='black')

            if 'iqr' in methods:
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                axes[1, i].axvline(lower_bound, color='red', linestyle='--', alpha=0.7, label='IQR bounds')
                axes[1, i].axvline(upper_bound, color='red', linestyle='--', alpha=0.7)

            axes[1, i].set_title(f'Distribution: {col}', fontweight='bold')
            axes[1, i].grid(True, alpha=0.3)
            axes[1, i].legend()

        plt.suptitle('Outlier Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

        return pd.DataFrame(outlier_summary).T

    @staticmethod
    def data_quality_report(data):
        """
        Comprehensive data quality assessment.

        Args:
            data: pandas DataFrame
        """
        print("DATA QUALITY ASSESSMENT")
        print("="*50)

        quality_metrics = {}

        for col in data.columns:
            col_metrics = {}
            series = data[col]

            # Basic metrics
            col_metrics['dtype'] = str(series.dtype)
            col_metrics['non_null_count'] = series.count()
            col_metrics['null_count'] = series.isnull().sum()
            col_metrics['null_percentage'] = (col_metrics['null_count'] / len(series)) * 100
            col_metrics['unique_count'] = series.nunique()
            col_metrics['unique_percentage'] = (col_metrics['unique_count'] / len(series)) * 100

            # Data type specific metrics
            if pd.api.types.is_numeric_dtype(series):
                col_metrics['has_negative'] = (series < 0).any()
                col_metrics['has_zero'] = (series == 0).any()
                col_metrics['has_infinite'] = np.isinf(series).any()

                if series.dtype in ['int64', 'int32']:
                    col_metrics['potential_categorical'] = col_metrics['unique_count'] <= 20

            elif pd.api.types.is_object_dtype(series):
                col_metrics['avg_string_length'] = series.astype(str).str.len().mean()
                col_metrics['has_mixed_case'] = len(set(series.astype(str).str.lower()) - set(series.astype(str))) > 0
                col_metrics['has_leading_trailing_spaces'] = (series.astype(str) != series.astype(str).str.strip()).any()

                # Check for potential numeric data stored as strings
                try:
                    pd.to_numeric(series.dropna(), errors='raise')
                    col_metrics['potentially_numeric'] = True
                except:
                    col_metrics['potentially_numeric'] = False

            # Quality flags
            flags = []
            if col_metrics['null_percentage'] > 50:
                flags.append('HIGH_MISSING')
            if col_metrics['unique_count'] == 1:
                flags.append('CONSTANT')
            if col_metrics['unique_count'] == len(series):
                flags.append('UNIQUE_IDENTIFIER')
            if pd.api.types.is_numeric_dtype(series) and np.isinf(series).any():
                flags.append('HAS_INFINITE')

            col_metrics['quality_flags'] = flags
            quality_metrics[col] = col_metrics

        # Create summary
        quality_df = pd.DataFrame(quality_metrics).T

        # Print summary
        print(f"\nDataset Shape: {data.shape}")
        print(f"Total Missing Values: {data.isnull().sum().sum()}")
        print(f"Memory Usage: {data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

        # Columns with quality issues
        problematic_cols = quality_df[quality_df['quality_flags'].str.len() > 0]
        if not problematic_cols.empty:
            print(f"\nColumns with Quality Issues:")
            for col, row in problematic_cols.iterrows():
                print(f"  {col}: {', '.join(row['quality_flags'])}")

        # Missing data visualization
        if data.isnull().sum().sum() > 0:
            plt.figure(figsize=(12, 6))
            missing_data = data.isnull().sum().sort_values(ascending=False)
            missing_data = missing_data[missing_data > 0]

            sns.barplot(x=missing_data.values, y=missing_data.index, palette='viridis')
            plt.title('Missing Values by Column', fontsize=14, fontweight='bold')
            plt.xlabel('Number of Missing Values')
            plt.tight_layout()
            plt.show()

        return quality_df

    @staticmethod
    def feature_engineering_suggestions(data, target_column=None):
        """
        Provide feature engineering suggestions based on data analysis.

        Args:
            data: pandas DataFrame
            target_column: str, target variable name
        """
        suggestions = []

        # Analyze numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        if target_column in numeric_cols:
            numeric_cols.remove(target_column)

        # Suggest transformations for skewed data
        for col in numeric_cols:
            series = data[col].dropna()
            if len(series) > 0:
                skewness = stats.skew(series)
                if abs(skewness) > 1:
                    if skewness > 1:
                        suggestions.append(f"Apply log transformation to {col} (right-skewed, skewness: {skewness:.2f})")
                    else:
                        suggestions.append(f"Apply square transformation to {col} (left-skewed, skewness: {skewness:.2f})")

        # Suggest feature combinations
        if len(numeric_cols) >= 2:
            suggestions.append("Consider creating ratio features between numeric variables")
            suggestions.append("Consider polynomial features for non-linear relationships")

        # Analyze categorical columns
        categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
        if target_column in categorical_cols:
            categorical_cols.remove(target_column)

        for col in categorical_cols:
            unique_count = data[col].nunique()
            if unique_count > 50:
                suggestions.append(f"Consider grouping rare categories in {col} ({unique_count} unique values)")

        # Datetime suggestions
        datetime_cols = data.select_dtypes(include=['datetime64']).columns.tolist()
        if datetime_cols:
            suggestions.append("Extract datetime features: year, month, day, hour, day_of_week")
            suggestions.append("Consider creating time-based lag features")

        # Missing value suggestions
        missing_cols = data.columns[data.isnull().any()].tolist()
        if missing_cols:
            suggestions.append("Create 'is_missing' indicator variables for columns with missing values")

        print("FEATURE ENGINEERING SUGGESTIONS")
        print("="*50)
        for i, suggestion in enumerate(suggestions, 1):
            print(f"{i}. {suggestion}")

        return suggestions


# Usage examples:
"""
# Feature importance analysis
importance_df = EDAUtilities.feature_importance_analysis(data, 'target_column')

# Correlation analysis
high_corr_df = EDAUtilities.correlation_analysis(data, threshold=0.8)

# Distribution analysis
dist_df = EDAUtilities.distribution_analysis(data, test_normality=True)

# Categorical encoding analysis
encoding_df = EDAUtilities.categorical_encoding_analysis(data, target_column='target')

# Outlier analysis
outlier_df = EDAUtilities.outlier_treatment_analysis(data)

# Data quality report
quality_df = EDAUtilities.data_quality_report(data)

# Feature engineering suggestions
suggestions = EDAUtilities.feature_engineering_suggestions(data, 'target_column')
"""
