#!/usr/bin/env python3
"""
Create meaningful plots for Bayesian Network evaluation results
"""

import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, confusion_matrix
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_results():
    """Load evaluation results"""
    with open('evaluation_results.json', 'r') as f:
        results = json.load(f)
    
    # Also need to load the actual predictions for plotting
    # We'll need to re-run evaluation or save predictions
    # For now, let's create a script that loads from saved predictions if available
    return results

def plot_comparison_bar(results):
    """Plot bar chart comparing ML vs EM metrics"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    methods = ['ML', 'EM']
    ml_acc = results['ml']['accuracy']
    em_acc = results['em']['accuracy']
    ml_auc = results['ml']['auc']
    em_auc = results['em']['auc']
    
    # Accuracy comparison
    accuracies = [ml_acc, em_acc]
    colors = ['#e74c3c', '#2ecc71']
    bars1 = ax1.bar(methods, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title('Accuracy Comparison: ML vs EM', fontsize=14, fontweight='bold')
    ax1.set_ylim([0, 1])
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, label='Random (0.5)')
    
    # Add value labels on bars
    for i, (bar, acc) in enumerate(zip(bars1, accuracies)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.3f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add improvement arrow
    if em_acc > ml_acc:
        ax1.annotate('', xy=(1, em_acc), xytext=(0, ml_acc),
                    arrowprops=dict(arrowstyle='->', lw=2, color='green'))
        ax1.text(0.5, (ml_acc + em_acc)/2 + 0.05, f'+{em_acc - ml_acc:.1%}',
                ha='center', fontsize=10, fontweight='bold', color='green')
    
    ax1.legend()
    
    # AUC comparison
    aucs = [ml_auc, em_auc]
    bars2 = ax2.bar(methods, aucs, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('AUC-ROC', fontsize=12, fontweight='bold')
    ax2.set_title('AUC-ROC Comparison: ML vs EM', fontsize=14, fontweight='bold')
    ax2.set_ylim([0, 1])
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, label='Random (0.5)')
    
    # Add value labels on bars
    for i, (bar, auc) in enumerate(zip(bars2, aucs)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{auc:.3f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add improvement arrow
    if em_auc > ml_auc:
        ax2.annotate('', xy=(1, em_auc), xytext=(0, ml_auc),
                    arrowprops=dict(arrowstyle='->', lw=2, color='green'))
        ax2.text(0.5, (ml_auc + em_auc)/2 + 0.05, f'+{em_auc - ml_auc:.1%}',
                ha='center', fontsize=10, fontweight='bold', color='green')
    
    ax2.legend()
    
    plt.tight_layout()
    return fig

def plot_roc_curves(ml_probs, ml_labels, em_probs, em_labels):
    """Plot ROC curves for both models"""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # ML ROC curve
    if len(np.unique(ml_labels)) > 1:
        fpr_ml, tpr_ml, _ = roc_curve(ml_labels, ml_probs)
        auc_ml = np.trapz(tpr_ml, fpr_ml)
        ax.plot(fpr_ml, tpr_ml, label=f'ML (AUC = {auc_ml:.3f})', 
                linewidth=2.5, color='#e74c3c')
    else:
        ax.plot([0, 1], [0, 1], '--', label='ML (AUC = 0.500)', 
                linewidth=2.5, color='#e74c3c', alpha=0.5)
    
    # EM ROC curve
    if len(np.unique(em_labels)) > 1:
        fpr_em, tpr_em, _ = roc_curve(em_labels, em_probs)
        auc_em = np.trapz(tpr_em, fpr_em)
        ax.plot(fpr_em, tpr_em, label=f'EM (AUC = {auc_em:.3f})', 
                linewidth=2.5, color='#2ecc71')
    else:
        ax.plot([0, 1], [0, 1], '--', label='EM (AUC = 0.500)', 
                linewidth=2.5, color='#2ecc71', alpha=0.5)
    
    # Random classifier line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5, label='Random (AUC = 0.500)')
    
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title('ROC Curves: ML vs EM', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    return fig

def plot_confusion_matrices(ml_preds, ml_labels, em_preds, em_labels):
    """Plot confusion matrices for both models"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # ML confusion matrix
    cm_ml = confusion_matrix(ml_labels, ml_preds)
    sns.heatmap(cm_ml, annot=True, fmt='d', cmap='Reds', ax=ax1,
                cbar_kws={'label': 'Count'}, square=True, linewidths=1)
    ax1.set_xlabel('Predicted', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Actual', fontsize=11, fontweight='bold')
    ax1.set_title('ML Confusion Matrix', fontsize=13, fontweight='bold')
    ax1.set_xticklabels(['Low Response', 'High Response'])
    ax1.set_yticklabels(['Low Response', 'High Response'])
    
    # Calculate accuracy for display
    acc_ml = np.trace(cm_ml) / np.sum(cm_ml)
    ax1.text(0.5, -0.15, f'Accuracy: {acc_ml:.1%}', 
            transform=ax1.transAxes, ha='center', fontsize=10, fontweight='bold')
    
    # EM confusion matrix
    cm_em = confusion_matrix(em_labels, em_preds)
    sns.heatmap(cm_em, annot=True, fmt='d', cmap='Greens', ax=ax2,
                cbar_kws={'label': 'Count'}, square=True, linewidths=1)
    ax2.set_xlabel('Predicted', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Actual', fontsize=11, fontweight='bold')
    ax2.set_title('EM Confusion Matrix', fontsize=13, fontweight='bold')
    ax2.set_xticklabels(['Low Response', 'High Response'])
    ax2.set_yticklabels(['Low Response', 'High Response'])
    
    # Calculate accuracy for display
    acc_em = np.trace(cm_em) / np.sum(cm_em)
    ax2.text(0.5, -0.15, f'Accuracy: {acc_em:.1%}', 
            transform=ax2.transAxes, ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    return fig

def plot_prediction_distributions(ml_probs, em_probs, ml_labels, em_labels):
    """Plot distribution of predicted probabilities"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # ML predictions
    ax1.hist(ml_probs[ml_labels == 0], bins=20, alpha=0.6, label='Low Response (Actual)', 
            color='#e74c3c', edgecolor='black', linewidth=1)
    ax1.hist(ml_probs[ml_labels == 1], bins=20, alpha=0.6, label='High Response (Actual)', 
            color='#3498db', edgecolor='black', linewidth=1)
    ax1.axvline(x=0.5, color='gray', linestyle='--', linewidth=1.5, label='Decision Threshold')
    ax1.set_xlabel('Predicted Probability (P(High Response))', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax1.set_title('ML: Predicted Probability Distribution', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3, linestyle='--', axis='y')
    
    # EM predictions
    ax2.hist(em_probs[em_labels == 0], bins=20, alpha=0.6, label='Low Response (Actual)', 
            color='#e74c3c', edgecolor='black', linewidth=1)
    ax2.hist(em_probs[em_labels == 1], bins=20, alpha=0.6, label='High Response (Actual)', 
            color='#3498db', edgecolor='black', linewidth=1)
    ax2.axvline(x=0.5, color='gray', linestyle='--', linewidth=1.5, label='Decision Threshold')
    ax2.set_xlabel('Predicted Probability (P(High Response))', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax2.set_title('EM: Predicted Probability Distribution', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3, linestyle='--', axis='y')
    
    plt.tight_layout()
    return fig

def main():
    # Load results
    results = load_results()
    
    # We need to re-run evaluation to get predictions for plotting
    # Or load from saved file if available
    print("Loading predictions from evaluation...")
    
    # Re-import evaluation to get predictions
    import sys
    sys.path.insert(0, '.')
    from evaluate import evaluate_model
    
    test_data_dir = "/Users/GayathriRajesh/Desktop/UCSD 27/Fall 2025/250A/project/Palbociclib_test"
    
    # Get predictions for both models
    print("Getting ML predictions...")
    ml_results = evaluate_model(
        "cpts_ml_hidden_pathways.json",
        test_data_dir,
        "ML (Maximum Likelihood)"
    )
    
    print("Getting EM predictions...")
    em_results = evaluate_model(
        "cpts_em_hidden_pathways.json",
        test_data_dir,
        "EM (Expectation Maximization)"
    )
    
    if ml_results and em_results:
        # Create all plots
        print("\nGenerating plots...")
        
        # 1. Comparison bar chart
        fig1 = plot_comparison_bar(results)
        fig1.savefig('evaluation_comparison.png', dpi=300, bbox_inches='tight')
        print("  Saved: evaluation_comparison.png")
        
        # 2. ROC curves
        fig2 = plot_roc_curves(
            ml_results['probabilities'], ml_results['true_labels'],
            em_results['probabilities'], em_results['true_labels']
        )
        fig2.savefig('evaluation_roc_curves.png', dpi=300, bbox_inches='tight')
        print("  Saved: evaluation_roc_curves.png")
        
        # 3. Confusion matrices
        fig3 = plot_confusion_matrices(
            ml_results['predictions'], ml_results['true_labels'],
            em_results['predictions'], em_results['true_labels']
        )
        fig3.savefig('evaluation_confusion_matrices.png', dpi=300, bbox_inches='tight')
        print("  Saved: evaluation_confusion_matrices.png")
        
        # 4. Prediction distributions
        fig4 = plot_prediction_distributions(
            ml_results['probabilities'], em_results['probabilities'],
            ml_results['true_labels'], em_results['true_labels']
        )
        fig4.savefig('evaluation_prediction_distributions.png', dpi=300, bbox_inches='tight')
        print("  Saved: evaluation_prediction_distributions.png")
        
        # Create a comprehensive figure with all plots
        fig_all = plt.figure(figsize=(16, 12))
        
        
        # 2b. Save AUC-only ROC curve as separate figure
        fig_auc = plt.figure(figsize=(8, 8))
        ax_auc = fig_auc.add_subplot(111)

        # ML ROC
        fpr_ml, tpr_ml, _ = roc_curve(ml_results['true_labels'], ml_results['probabilities'])
        auc_ml = np.trapz(tpr_ml, fpr_ml)
        ax_auc.plot(fpr_ml, tpr_ml, label=f'ML (AUC = {auc_ml:.3f})', linewidth=2.5, color='#e74c3c')

        # EM ROC
        fpr_em, tpr_em, _ = roc_curve(em_results['true_labels'], em_results['probabilities'])
        auc_em = np.trapz(tpr_em, fpr_em)
        ax_auc.plot(fpr_em, tpr_em, label=f'EM (AUC = {auc_em:.3f})', linewidth=2.5, color='#2ecc71')

        # Random line
        ax_auc.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5, label='Random (AUC = 0.500)')

        ax_auc.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        ax_auc.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        ax_auc.set_title('AUC–ROC Curve', fontsize=14, fontweight='bold')
        ax_auc.legend(loc='lower right', fontsize=11)
        ax_auc.grid(alpha=0.3, linestyle='--')
        ax_auc.set_xlim([0, 1])
        ax_auc.set_ylim([0, 1])

        fig_auc.tight_layout()
        fig_auc.savefig('evaluation_auc_only.png', dpi=300, bbox_inches='tight')
        print("  Saved: evaluation_auc_only.png")

        
        # Top row: Comparison and ROC
        ax1 = plt.subplot(2, 3, 1)
        methods = ['ML', 'EM']
        accuracies = [results['ml']['accuracy'], results['em']['accuracy']]
        colors = ['#e74c3c', '#2ecc71']
        bars = ax1.bar(methods, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax1.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
        ax1.set_title('Accuracy Comparison', fontsize=12, fontweight='bold')
        ax1.set_ylim([0, 1])
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        ax1.axhline(y=0.5, color='gray', linestyle='--', linewidth=1)
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax2 = plt.subplot(2, 3, 2)
        aucs = [results['ml']['auc'], results['em']['auc']]
        bars = ax2.bar(methods, aucs, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax2.set_ylabel('AUC-ROC', fontsize=11, fontweight='bold')
        ax2.set_title('AUC-ROC Comparison', fontsize=12, fontweight='bold')
        ax2.set_ylim([0, 1])
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        ax2.axhline(y=0.5, color='gray', linestyle='--', linewidth=1)
        for bar, auc in zip(bars, aucs):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{auc:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax3 = plt.subplot(2, 3, 3)
        fpr_em, tpr_em, _ = roc_curve(em_results['true_labels'], em_results['probabilities'])
        ax3.plot(fpr_em, tpr_em, label=f'EM (AUC = {results["em"]["auc"]:.3f})', 
                linewidth=2.5, color='#2ecc71')
        if len(np.unique(ml_results['true_labels'])) > 1:
            fpr_ml, tpr_ml, _ = roc_curve(ml_results['true_labels'], ml_results['probabilities'])
            ax3.plot(fpr_ml, tpr_ml, label=f'ML (AUC = {results["ml"]["auc"]:.3f})', 
                    linewidth=2.5, color='#e74c3c', alpha=0.7)
        ax3.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5, label='Random')
        ax3.set_xlabel('False Positive Rate', fontsize=10, fontweight='bold')
        ax3.set_ylabel('True Positive Rate', fontsize=10, fontweight='bold')
        ax3.set_title('ROC Curves', fontsize=12, fontweight='bold')
        ax3.legend(fontsize=9)
        ax3.grid(alpha=0.3, linestyle='--')
        
        # Bottom row: Confusion matrices
        ax4 = plt.subplot(2, 3, 4)
        cm_ml = confusion_matrix(ml_results['true_labels'], ml_results['predictions'])
        sns.heatmap(cm_ml, annot=True, fmt='d', cmap='Reds', ax=ax4, cbar=False, square=True)
        ax4.set_xlabel('Predicted', fontsize=10, fontweight='bold')
        ax4.set_ylabel('Actual', fontsize=10, fontweight='bold')
        ax4.set_title(f'ML Confusion Matrix\n(Accuracy: {results["ml"]["accuracy"]:.1%})', 
                     fontsize=11, fontweight='bold')
        ax4.set_xticklabels(['Low', 'High'])
        ax4.set_yticklabels(['Low', 'High'])
        
        ax5 = plt.subplot(2, 3, 5)
        cm_em = confusion_matrix(em_results['true_labels'], em_results['predictions'])
        sns.heatmap(cm_em, annot=True, fmt='d', cmap='Greens', ax=ax5, cbar=False, square=True)
        ax5.set_xlabel('Predicted', fontsize=10, fontweight='bold')
        ax5.set_ylabel('Actual', fontsize=10, fontweight='bold')
        ax5.set_title(f'EM Confusion Matrix\n(Accuracy: {results["em"]["accuracy"]:.1%})', 
                     fontsize=11, fontweight='bold')
        ax5.set_xticklabels(['Low', 'High'])
        ax5.set_yticklabels(['Low', 'High'])
        
        # Prediction distribution for EM (the better model)
        ax6 = plt.subplot(2, 3, 6)
        ax6.hist(em_results['probabilities'][em_results['true_labels'] == 0], 
                bins=15, alpha=0.6, label='Low Response', color='#e74c3c', edgecolor='black')
        ax6.hist(em_results['probabilities'][em_results['true_labels'] == 1], 
                bins=15, alpha=0.6, label='High Response', color='#3498db', edgecolor='black')
        ax6.axvline(x=0.5, color='gray', linestyle='--', linewidth=1.5)
        ax6.set_xlabel('Predicted Probability', fontsize=10, fontweight='bold')
        ax6.set_ylabel('Frequency', fontsize=10, fontweight='bold')
        ax6.set_title('EM: Probability Distribution', fontsize=11, fontweight='bold')
        ax6.legend(fontsize=9)
        ax6.grid(alpha=0.3, linestyle='--', axis='y')
        
        plt.suptitle('Bayesian Network Evaluation: ML vs EM on Test Set', 
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        fig_all.savefig('evaluation_comprehensive.png', dpi=300, bbox_inches='tight')
        print("  Saved: evaluation_comprehensive.png")
        
        print("\nAll plots generated successfully!")
        plt.close('all')
    else:
        print("ERROR: Could not get predictions for plotting")

if __name__ == "__main__":
    main()


