"""
Motor Density Analysis Pipeline

This file is used to go through the images in data and apply the motor density estimator script on them.
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

try:
    from motor_density_estimator import MotorDensityEstimator
except ImportError:
    print("Error: motor_density_estimator.py not found in current directory")
    exit(1)

class DataAnalysisPipeline:
    def __init__(self, data_root: str = None, output_dir: str = None):
        # Get the base directory (parent of src folder)
        BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        
        # Set default paths relative to base directory
        if data_root is None:
            data_root = os.path.join(BASE_DIR, 'data')
        if output_dir is None:
            output_dir = os.path.join(BASE_DIR, 'analysis_results2')
            
        self.data_root = Path(data_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create output directories
        (self.output_dir / "visualizations").mkdir(exist_ok=True)
        (self.output_dir / "processed_data").mkdir(exist_ok=True)
        (self.output_dir / "sample_results").mkdir(exist_ok=True)
        
        self.estimator = MotorDensityEstimator(enable_debug=False)
        self.results = []
        
    def discover_data_structure(self) -> Dict:
        """Find and count images in data folders"""
        structure = {
            'old_data': {'path': self.data_root / 'raw' / 'old', 'images': []},
            'new_data': {'path': self.data_root / 'raw' / 'new', 'categories': {}}
        }
        
        print(f"Looking for data in: {self.data_root}")
        print(f"Old data path: {structure['old_data']['path']}")
        print(f"New data path: {structure['new_data']['path']}")
        
        # Process old data folder
        old_path = structure['old_data']['path']
        if old_path.exists():
            structure['old_data']['images'] = [
                f for f in os.listdir(old_path) 
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ]
            print(f"Found {len(structure['old_data']['images'])} images in old data folder")
        else:
            print(f"Old data folder does not exist: {old_path}")
        
        # Process new data categories
        new_path = structure['new_data']['path']
        if new_path.exists():
            print(f"Scanning new data folder: {new_path}")
            for category_dir in os.listdir(new_path):
                category_path = new_path / category_dir
                if category_path.is_dir():
                    images = [
                        f for f in os.listdir(category_path)
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
                    ]
                    structure['new_data']['categories'][category_dir] = {
                        'path': category_path,
                        'images': images,
                        'count': len(images)
                    }
        else:
            print(f"New data folder does not exist: {new_path}")
        
        return structure
    
    def process_single_image(self, image_path: Path, category: str = None, dataset: str = None) -> Dict:
        """Process one image and extract density metrics"""
        try:
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                return {'error': 'Could not load image', 'path': str(image_path)}
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Run density estimation
            results = self.estimator.estimate_density(image_rgb)
            
            # Package results
            return {
                'image_path': str(image_path),
                'filename': image_path.name,
                'category': category,
                'dataset': dataset,
                'density': results['density'],
                'hsv_density': results['individual_densities']['hsv'],
                'lab_density': results['individual_densities']['lab'],
                'adaptive_density': results['individual_densities']['adaptive'],
                'processing_time': results['metadata']['processing_time_seconds'],
                'roi_coverage': results['metadata']['roi_coverage'],
                'image_shape': results['metadata']['image_shape'],
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'path': str(image_path),
                'category': category,
                'dataset': dataset
            }
    
    def process_dataset(self, max_images_per_category: int = 50) -> pd.DataFrame:
        """Process all images in the dataset"""
        print("Discovering data structure...")
        structure = self.discover_data_structure()
        
        # Print summary
        old_count = len(structure['old_data']['images'])
        new_categories = structure['new_data']['categories']
        new_total = sum(cat['count'] for cat in new_categories.values())
        
        print(f"Data Summary:")
        print(f"   Old dataset: {old_count} images")
        print(f"   New dataset: {len(new_categories)} categories, {new_total} total images")
        for cat_name, cat_info in new_categories.items():
            print(f"     - {cat_name}: {cat_info['count']} images")
        
        results = []
        total_processed = 0
        
        # Process old data
        print(f"\nProcessing old dataset...")
        old_images = structure['old_data']['images'][:max_images_per_category]
        for i, filename in enumerate(old_images):
            image_path = structure['old_data']['path'] / filename
            result = self.process_single_image(image_path, category='old_data', dataset='old')
            results.append(result)
            total_processed += 1
            
            if (i + 1) % 10 == 0:
                print(f"   Processed {i + 1}/{len(old_images)} old images")
        
        # Process new data by category
        for category_name, category_info in new_categories.items():
            print(f"\nProcessing category: {category_name}")
            images_to_process = category_info['images'][:max_images_per_category]
            
            for i, filename in enumerate(images_to_process):
                image_path = category_info['path'] / filename
                result = self.process_single_image(image_path, category=category_name, dataset='new')
                results.append(result)
                total_processed += 1
                
                if (i + 1) % 10 == 0:
                    print(f"   Processed {i + 1}/{len(images_to_process)} images in {category_name}")
        
        print(f"\nTotal images processed: {total_processed}")
        
        # Convert to DataFrame and save
        df = pd.DataFrame(results)
        
        # Check if we have any data
        if len(df) == 0:
            print("ERROR: No images were found or processed!")
            print("Please check:")
            print(f"  - Data path exists: {self.data_root}")
            print(f"  - Old data path: {self.data_root / 'raw' / 'old'}")
            print(f"  - New data path: {self.data_root / 'raw' / 'new'}")
            return pd.DataFrame()
        
        df.to_csv(self.output_dir / 'processed_data' / 'all_results.csv', index=False)
        
        # Separate successful results from errors
        if 'density' in df.columns:
            df_clean = df[~df['density'].isna()].copy()
            errors_df = df[df['density'].isna()].copy()
        else:
            print("ERROR: No successful processing occurred!")
            return pd.DataFrame()
        
        if len(errors_df) > 0:
            print(f"{len(errors_df)} images had errors (saved to errors.csv)")
            errors_df.to_csv(self.output_dir / 'processed_data' / 'errors.csv', index=False)
        
        print(f"Successfully processed {len(df_clean)} images for analysis")
        return df_clean
    
    def generate_analysis_report(self, df: pd.DataFrame):
        """Generate comprehensive analysis and visualizations"""
        print("\nGenerating analysis report...")
        
        # Calculate basic statistics
        stats = {
            'total_images': len(df),
            'categories': df['category'].unique().tolist(),
            'datasets': df['dataset'].unique().tolist(),
            'density_stats': {
                'mean': float(df['density'].mean()),
                'median': float(df['density'].median()),
                'std': float(df['density'].std()),
                'min': float(df['density'].min()),
                'max': float(df['density'].max()),
                'q25': float(df['density'].quantile(0.25)),
                'q75': float(df['density'].quantile(0.75))
            },
            'processing_time_stats': {
                'mean_seconds': float(df['processing_time'].mean()),
                'total_hours': float(df['processing_time'].sum() / 3600)
            }
        }
        
        # Category-wise statistics
        category_stats = df.groupby('category').agg({
            'density': ['count', 'mean', 'std', 'min', 'max'],
            'hsv_density': 'mean',
            'lab_density': 'mean', 
            'adaptive_density': 'mean',
            'processing_time': 'mean'
        }).round(3)
        
        # Save statistics
        with open(self.output_dir / 'processed_data' / 'summary_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        category_stats.to_csv(self.output_dir / 'processed_data' / 'category_stats.csv')
        
        # Generate visualizations
        self.create_visualizations(df)
        
        # Generate text report
        self.create_text_report(df, stats)
        
        print(f"Analysis complete! Results saved to {self.output_dir}")
    
    def create_visualizations(self, df: pd.DataFrame):
        """Create visualizations optimized for large datasets"""
        plt.style.use('default')
        
        # Main overview figure
        plt.figure(figsize=(15, 10))
        
        # Overall density distribution
        plt.subplot(2, 3, 1)
        plt.hist(df['density'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title('Overall Density Distribution')
        plt.xlabel('Density')
        plt.ylabel('Frequency')
        plt.axvline(df['density'].mean(), color='red', linestyle='--', label=f'Mean: {df["density"].mean():.3f}')
        plt.legend()
        
        # Top 10 categories by count
        plt.subplot(2, 3, 2)
        top_categories = df['category'].value_counts().head(10)
        top_categories.plot(kind='bar')
        plt.title('Top 10 Categories by Image Count')
        plt.xlabel('Category')
        plt.ylabel('Image Count')
        plt.xticks(rotation=45, ha='right')
        
        # Method comparison
        plt.subplot(2, 3, 3)
        method_data = [df['hsv_density'], df['lab_density'], df['adaptive_density'], df['density']]
        method_labels = ['HSV', 'LAB', 'Adaptive', 'Final\n(Ensemble)']
        plt.boxplot(method_data, labels=method_labels)
        plt.title('Method Performance Comparison')
        plt.ylabel('Density')
        
        # Density by dataset
        plt.subplot(2, 3, 4)
        df.boxplot(column='density', by='dataset', ax=plt.gca())
        plt.title('Density by Dataset')
        plt.suptitle('')
        
        # Top 15 categories by average density
        plt.subplot(2, 3, 5)
        category_means = df.groupby('category')['density'].mean().sort_values(ascending=False).head(15)
        category_means.plot(kind='bar')
        plt.title('Top 15 Categories by Average Density')
        plt.xlabel('Category')
        plt.ylabel('Average Density')
        plt.xticks(rotation=45, ha='right')
        
        # Processing time vs image count
        plt.subplot(2, 3, 6)
        category_stats = df.groupby('category').agg({
            'processing_time': 'mean',
            'density': 'count'
        }).rename(columns={'density': 'image_count'})
        
        scatter = plt.scatter(category_stats['image_count'], 
                            category_stats['processing_time'],
                            alpha=0.6, s=50)
        plt.xlabel('Images per Category')
        plt.ylabel('Average Processing Time (s)')
        plt.title('Processing Time vs Category Size')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'visualizations' / 'overview.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Correlation heatmap
        plt.figure(figsize=(10, 8))
        corr_columns = ['density', 'hsv_density', 'lab_density', 'adaptive_density', 'roi_coverage', 'processing_time']
        correlation_matrix = df[corr_columns].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, linewidths=0.5)
        plt.title('Method Correlation Matrix')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'visualizations' / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Top categories analysis
        top_10_categories = df['category'].value_counts().head(10).index
        df_top = df[df['category'].isin(top_10_categories)]
        
        if len(df_top) > 0:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # Average density for top categories
            category_means = df_top.groupby('category')['density'].mean().sort_values(ascending=False)
            category_means.plot(kind='bar', ax=axes[0,0], color='steelblue')
            axes[0,0].set_title('Average Density - Top 10 Categories')
            axes[0,0].set_ylabel('Average Density')
            axes[0,0].tick_params(axis='x', rotation=45)
            
            # Method performance for top categories
            method_performance = df_top.groupby('category')[['hsv_density', 'lab_density', 'adaptive_density']].mean()
            method_performance.plot(kind='bar', ax=axes[0,1], width=0.8)
            axes[0,1].set_title('Method Performance - Top 10 Categories')
            axes[0,1].set_ylabel('Average Density')
            axes[0,1].tick_params(axis='x', rotation=45)
            axes[0,1].legend(title='Method')
            
            # Density distribution for top categories
            df_top.boxplot(column='density', by='category', ax=axes[1,0])
            axes[1,0].set_title('Density Distribution - Top 10 Categories')
            axes[1,0].set_xlabel('Category')
            axes[1,0].set_ylabel('Density')
            axes[1,0].tick_params(axis='x', rotation=45)
            plt.suptitle('')
            
            # ROI coverage for top categories
            roi_means = df_top.groupby('category')['roi_coverage'].mean().sort_values(ascending=False)
            roi_means.plot(kind='bar', ax=axes[1,1], color='orange')
            axes[1,1].set_title('Average ROI Coverage - Top 10 Categories')
            axes[1,1].set_ylabel('ROI Coverage Ratio')
            axes[1,1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'visualizations' / 'top_categories_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # Method comparison analysis
        plt.figure(figsize=(15, 5))
        
        # Which method wins by category
        plt.subplot(1, 3, 1)
        method_wins = []
        for category in df['category'].unique():
            cat_data = df[df['category'] == category]
            method_means = {
                'HSV': cat_data['hsv_density'].mean(),
                'LAB': cat_data['lab_density'].mean(), 
                'Adaptive': cat_data['adaptive_density'].mean()
            }
            best_method = max(method_means, key=method_means.get)
            method_wins.append(best_method)
        
        method_win_counts = pd.Series(method_wins).value_counts()
        method_win_counts.plot(kind='pie', ax=plt.gca(), autopct='%1.1f%%')
        plt.title('Best Method by Category Count')
        plt.ylabel('')
        
        # Overall method performance
        plt.subplot(1, 3, 2)
        overall_performance = df[['hsv_density', 'lab_density', 'adaptive_density', 'density']].mean()
        overall_performance.plot(kind='bar', color=['red', 'green', 'blue', 'purple'])
        plt.title('Overall Method Performance')
        plt.ylabel('Average Density')
        plt.xticks(rotation=45)
        
        # Method consistency
        plt.subplot(1, 3, 3)
        method_variance = df[['hsv_density', 'lab_density', 'adaptive_density', 'density']].std()
        method_variance.plot(kind='bar', color=['red', 'green', 'blue', 'purple'])
        plt.title('Method Consistency (Lower = More Consistent)')
        plt.ylabel('Standard Deviation')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'visualizations' / 'method_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_text_report(self, df: pd.DataFrame, stats: Dict):
        """Generate comprehensive text report"""
        report_path = self.output_dir / 'ANALYSIS_REPORT.txt'
        
        with open(report_path, 'w') as f:
            f.write("MOTOR DENSITY ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Overall summary
            f.write("OVERALL SUMMARY\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total images processed: {stats['total_images']}\n")
            f.write(f"Categories found: {len(stats['categories'])}\n")
            f.write(f"Datasets: {', '.join(stats['datasets'])}\n\n")
            
            # Density statistics
            f.write("DENSITY STATISTICS\n")
            f.write("-" * 20 + "\n")
            density_stats = stats['density_stats']
            f.write(f"Mean density: {density_stats['mean']:.3f}\n")
            f.write(f"Median density: {density_stats['median']:.3f}\n")
            f.write(f"Standard deviation: {density_stats['std']:.3f}\n")
            f.write(f"Range: {density_stats['min']:.3f} - {density_stats['max']:.3f}\n")
            f.write(f"Interquartile range: {density_stats['q25']:.3f} - {density_stats['q75']:.3f}\n\n")
            
            # Category breakdown
            f.write("CATEGORY BREAKDOWN\n")
            f.write("-" * 20 + "\n")
            for category in df['category'].unique():
                cat_data = df[df['category'] == category]
                f.write(f"\n{category.upper()}:\n")
                f.write(f"  Images: {len(cat_data)}\n")
                f.write(f"  Average density: {cat_data['density'].mean():.3f}\n")
                f.write(f"  Density range: {cat_data['density'].min():.3f} - {cat_data['density'].max():.3f}\n")
                f.write(f"  Best method: ")
                
                # Find best method for this category
                method_means = {
                    'HSV': cat_data['hsv_density'].mean(),
                    'LAB': cat_data['lab_density'].mean(),
                    'Adaptive': cat_data['adaptive_density'].mean()
                }
                best_method = max(method_means, key=method_means.get)
                f.write(f"{best_method} ({method_means[best_method]:.3f})\n")
            
            # Performance metrics
            f.write(f"\nPERFORMANCE METRICS\n")
            f.write("-" * 20 + "\n")
            f.write(f"Average processing time: {stats['processing_time_stats']['mean_seconds']:.2f} seconds\n")
            f.write(f"Total processing time: {stats['processing_time_stats']['total_hours']:.2f} hours\n")
            f.write(f"Average ROI coverage: {df['roi_coverage'].mean():.3f}\n\n")
            
            # Recommendations
            f.write("RECOMMENDATIONS\n")
            f.write("-" * 20 + "\n")
            f.write("Based on the analysis:\n\n")
            
            if density_stats['mean'] > 0.8:
                f.write("High average density detected - pipeline working well for dense motor piles\n")
            elif density_stats['mean'] < 0.5:
                f.write("Low average density - may need parameter tuning for sparse piles\n")
            else:
                f.write("Moderate density range - pipeline performing as expected\n")
            
            if df['roi_coverage'].mean() < 0.7:
                f.write("Low ROI coverage - consider adjusting background masking parameters\n")
            
            f.write("Review category-specific results for optimal method selection\n")
            f.write("Consider the visualizations for deeper insights\n")
        
        print(f"Detailed report saved to {report_path}")
    
    def save_sample_results(self, df: pd.DataFrame, n_samples: int = 5):
        """Save sample processed images for visual inspection"""
        print(f"\nSaving {n_samples} sample results per category...")
        
        for category in df['category'].unique():
            category_data = df[df['category'] == category]
            
            # Get samples across density range
            high_density = category_data.nlargest(n_samples//2, 'density')
            low_density = category_data.nsmallest(n_samples//2, 'density')
            samples = pd.concat([high_density, low_density]).head(n_samples)
            
            category_dir = self.output_dir / 'sample_results' / category
            category_dir.mkdir(exist_ok=True)
            
            for _, row in samples.iterrows():
                # Load and process image
                image_path = Path(row['image_path'])
                if image_path.exists():
                    image = cv2.imread(str(image_path))
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # Get detailed results with masks
                    results = self.estimator.estimate_density(image_rgb)
                    
                    # Create visualization
                    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                    
                    # Original image
                    axes[0,0].imshow(image_rgb)
                    axes[0,0].set_title('Original Image')
                    axes[0,0].axis('off')
                    
                    # Final mask
                    axes[0,1].imshow(results['masks']['final'], cmap='gray')
                    axes[0,1].set_title(f'Final Result (Density: {results["density"]:.3f})')
                    axes[0,1].axis('off')
                    
                    # ROI mask
                    axes[1,0].imshow(results['masks']['roi'], cmap='gray')
                    axes[1,0].set_title('ROI Mask')
                    axes[1,0].axis('off')
                    
                    # Method comparison
                    method_densities = [
                        results['individual_densities']['hsv'],
                        results['individual_densities']['lab'],
                        results['individual_densities']['adaptive'],
                        results['density']
                    ]
                    method_labels = ['HSV', 'LAB', 'Adaptive', 'Final']
                    axes[1,1].bar(method_labels, method_densities)
                    axes[1,1].set_title('Method Comparison')
                    axes[1,1].set_ylabel('Density')
                    
                    plt.suptitle(f'{category} - {image_path.name}')
                    plt.tight_layout()
                    
                    # Save
                    output_path = category_dir / f"{image_path.stem}_analysis.png"
                    plt.savefig(output_path, dpi=150, bbox_inches='tight')
                    plt.close()

def main():
    """Main execution function"""
    print("Starting Motor Density Analysis")
    print("=" * 50)
    
    # Initialize pipeline
    pipeline = DataAnalysisPipeline()
    
    # Process all data
    df = pipeline.process_dataset(max_images_per_category=100)
    
    # Check if we have data to analyze
    if len(df) == 0:
        print("\nNo data to analyze. Exiting.")
        return
    
    # Generate analysis
    pipeline.generate_analysis_report(df)
    
    # Save sample results
    pipeline.save_sample_results(df)
    
    print("\nAnalysis complete!")
if __name__ == "__main__":
    main()