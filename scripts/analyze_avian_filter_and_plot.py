#!/usr/bin/env python3
"""
scripts/analyze_avian_filter_and_plot.py
----------------------------------------
Generates:
1. Publication-quality heatmaps of CCT p-values and selection detection density (before vs after filtering).
2. Detailed alignment context figure for exemplary artifact genes showing raw vs cleaned sequences.
3. Unbiased KEGG pathway enrichment analysis before vs after filtering.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import fisher_exact

plt.rcParams['font.sans-serif'] = 'Helvetica', 'Arial', 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8

def plot_heatmaps(df, out_path="hyphaeon_paper/figures/avian_filter_cct_density_heatmaps.pdf"):
    print("[*] Generating CCT p-value and selection density comparison heatmaps...")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)
    
    # 1. 2D Density / Scatter of -log10(p_Cauchy) Before vs After Filtering
    ax1 = axes[0]
    sns.kdeplot(
        data=df,
        x="neglog10_cct_raw",
        y="neglog10_cct_clean",
        cmap="mako",
        fill=True,
        thresh=0.02,
        levels=15,
        ax=ax1,
        cbar=True,
        cbar_kws={'label': 'Gene Density'}
    )
    # Also scatter the artifact genes
    art_df = df[df['artifacts_masked'] > 0]
    ax1.scatter(art_df['neglog10_cct_raw'], art_df['neglog10_cct_clean'], s=8, color='#e74c3c', alpha=0.35, label=f'Masked Artifact Genes (N = {len(art_df):,})')
    
    max_val = max(df['neglog10_cct_raw'].max(), df['neglog10_cct_clean'].max())
    ax1.plot([0, max_val], [0, max_val], '--', color='#7f8c8d', lw=1.5, label='Identity (Unchanged)')
    
    n_shifted = (df['delta_cct_neglog'] > 0.05).sum()
    ax1.text(0.05, 0.85, f"Genes with Extinguished\nArtifact Spikes: N = {n_shifted:,}\n({n_shifted/len(df)*100:.1f}%)", 
             transform=ax1.transAxes, fontsize=10, bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='#cccccc'))
    
    ax1.set_xlabel(r"Raw Baseline $-\log_{10}(p_{\mathrm{Cauchy}})$", fontsize=12)
    ax1.set_ylabel(r"Cleaned $-\log_{10}(p_{\mathrm{Cauchy}})$ (After Error Filter)", fontsize=12)
    ax1.set_title("A. Gene-Level Selection Significance (CCT)", fontsize=13, pad=10)
    ax1.legend(loc='lower right', frameon=True, fontsize=9)
    
    # 2. 2D Density / Scatter of Sitewise Selection Density (p <= 0.05) Before vs After
    ax2 = axes[1]
    sns.kdeplot(
        data=df,
        x=df["density_p05_raw"] * 100,
        y=df["density_p05_clean"] * 100,
        cmap="rocket",
        fill=True,
        thresh=0.02,
        levels=15,
        ax=ax2,
        cbar=True,
        cbar_kws={'label': 'Gene Density'}
    )
    ax2.scatter(art_df['density_p05_raw']*100, art_df['density_p05_clean']*100, s=8, color='#3498db', alpha=0.35, label=f'Masked Artifact Genes (N = {len(art_df):,})')
    
    max_d = max((df['density_p05_raw']*100).max(), (df['density_p05_clean']*100).max())
    ax2.plot([0, max_d], [0, max_d], '--', color='#7f8c8d', lw=1.5, label='Identity (Unchanged)')
    
    total_sites_extinguished = df['delta_sig_p05'].sum()
    ax2.text(0.05, 0.85, f"Total Spurious Sites\nExtinguished: N = {total_sites_extinguished:,}", 
             transform=ax2.transAxes, fontsize=10, bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='#cccccc'))
    
    ax2.set_xlabel(r"Raw Selected Codon Density (% Sites with $p \leq 0.05$)", fontsize=12)
    ax2.set_ylabel(r"Cleaned Selected Codon Density (%)", fontsize=12)
    ax2.set_title("B. Sitewise Positive Selection Density", fontsize=13, pad=10)
    ax2.legend(loc='lower right', frameon=True, fontsize=9)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches='tight')
    plt.savefig(out_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"[✓] Heatmaps saved to: {out_path} and .png")

def plot_alignment_context_figure(audit_df, out_path="hyphaeon_paper/figures/avian_artifact_alignment_context.pdf"):
    print("[*] Generating alignment context visualization for exemplary artifact genes...")
    # Find top 3 distinctive artifacts
    top_artifacts = audit_df.sort_values(by=['consecutive_mismatches', 'oci'], ascending=False).drop_duplicates(subset=['hog']).head(3)
    
    fig, axes = plt.subplots(len(top_artifacts), 1, figsize=(14, 3.2 * len(top_artifacts)), dpi=300)
    if len(top_artifacts) == 1: axes = [axes]
    
    for idx, (_, row) in enumerate(top_artifacts.iterrows()):
        ax = axes[idx]
        hog = row['hog']
        entrez = row.get('entrezgene', np.nan)
        entrez_str = f"Entrez {int(entrez)}" if pd.notna(entrez) else "Unannotated"
        taxon = row['outlier_taxon']
        start = int(row['patch_start_1idx'])
        end = int(row['patch_end_1idx'])
        con_seq = str(row['consensus_aa_seq'])
        obs_seq = str(row['outlier_aa_seq'])
        span = len(con_seq)
        
        ax.set_xlim(-6.5, max(span, 10) + 1.0)
        ax.set_ylim(-0.8, 2.8)
        ax.axis('off')
        
        title_str = f"HOG_{hog} ({entrez_str}) | Codons {start}–{end} | Isolated Frameshift Run in {taxon} ({row['consecutive_mismatches']} consec. muts, OCI = {row['oci']:.2f})"
        ax.text(-6.2, 2.4, title_str, fontsize=11, fontweight='bold', color='#2c3e50')
        
        ax.text(-6.2, 1.6, f"Consensus (38 species):", fontsize=9.5, fontfamily='monospace', color='#333333', va='center')
        ax.text(-6.2, 0.8, f"Outlier ({taxon}) [Raw]:", fontsize=9.5, fontfamily='monospace', color='#c0392b', va='center')
        ax.text(-6.2, 0.0, f"Outlier ({taxon}) [Masked]:", fontsize=9.5, fontfamily='monospace', color='#27ae60', va='center')
        
        for pos_i in range(span):
            c_char = con_seq[pos_i] if pos_i < len(con_seq) else '-'
            o_char = obs_seq[pos_i] if pos_i < len(obs_seq) else '-'
            is_mut = (c_char != o_char) and c_char not in '-?' and o_char not in '-?'
            
            # Consensus
            ax.text(pos_i + 0.5, 1.6, c_char, fontsize=11, fontfamily='monospace', ha='center', va='center', color='#2c3e50')
            
            # Raw Outlier
            box_col = '#fadbd8' if is_mut else '#f4f6f7'
            text_col = '#922b21' if is_mut else '#7f8c8d'
            ax.add_patch(plt.Rectangle((pos_i + 0.1, 0.45), 0.8, 0.7, facecolor=box_col, edgecolor='none', alpha=0.85))
            ax.text(pos_i + 0.5, 0.8, o_char, fontsize=11, fontfamily='monospace', ha='center', va='center', color=text_col, fontweight='bold' if is_mut else 'normal')
            
            # Masked Outlier
            m_box_col = '#d4efdf' if is_mut else '#f4f6f7'
            m_text_col = '#1e8449' if is_mut else '#7f8c8d'
            m_char = 'X' if is_mut else c_char
            ax.add_patch(plt.Rectangle((pos_i + 0.1, -0.35), 0.8, 0.7, facecolor=m_box_col, edgecolor='none', alpha=0.85))
            ax.text(pos_i + 0.5, 0.0, m_char, fontsize=11, fontfamily='monospace', ha='center', va='center', color=m_text_col, fontweight='bold' if is_mut else 'normal')
            
        ax.plot([-6.2, max(span, 10) + 0.5], [-0.55, -0.55], color='#d5dbdb', lw=1)
        
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches='tight')
    plt.savefig(out_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"[✓] Alignment context figure saved to: {out_path} and .png")

def run_kegg_enrichment_comparison(df, out_csv="hyphaeon_paper/tables/avian_kegg_enrichment_before_after.csv"):
    print("[*] Performing unbiased KEGG pathway over-representation analysis (Before vs After)...")
    kegg_ref_path = 'benchmark/avian_shultz2019/metadata/R_data_and_scripts/04_output_pathway_results/chicken_genetree_pathwayres_nocutoffs.csv'
    if not os.path.exists(kegg_ref_path):
        print("[!] KEGG reference file not found:", kegg_ref_path)
        return
        
    kegg_ref = pd.read_csv(kegg_ref_path)
    
    df_valid = df[df['entrezgene'].notna()].copy()
    df_valid['entrez_int'] = df_valid['entrezgene'].astype(int).astype(str)
    all_dataset_entrez = set(df_valid['entrez_int'])
    
    top_raw = set(df_valid.sort_values(by='cct_pval_raw', ascending=True).head(150)['entrez_int'])
    top_clean = set(df_valid.sort_values(by='cct_pval_clean', ascending=True).head(150)['entrez_int'])
    
    enrichment_records = []
    
    for _, row in kegg_ref.iterrows():
        p_id = row['ID']
        desc = row['Description']
        path_genes = set(str(row['geneID']).split('/')) & all_dataset_entrez
        n_path = len(path_genes)
        if n_path < 4: continue
        
        # Raw Fisher Test
        k_raw = len(top_raw & path_genes)
        K_raw = len(top_raw)
        N_tot = len(all_dataset_entrez)
        table_raw = [[k_raw, n_path - k_raw], [K_raw - k_raw, N_tot - n_path - (K_raw - k_raw)]]
        odds_raw, p_raw = fisher_exact(table_raw, alternative='greater')
        
        # Clean Fisher Test
        k_clean = len(top_clean & path_genes)
        K_clean = len(top_clean)
        table_clean = [[k_clean, n_path - k_clean], [K_clean - k_clean, N_tot - n_path - (K_clean - k_clean)]]
        odds_clean, p_clean = fisher_exact(table_clean, alternative='greater')
        
        enrichment_records.append({
            'KEGG_ID': p_id,
            'Pathway_Description': desc,
            'Path_Genes_in_Dataset': n_path,
            'Hits_Raw': k_raw,
            'P_Raw': p_raw,
            'OddsRatio_Raw': odds_raw,
            'Hits_Clean': k_clean,
            'P_Clean': p_clean,
            'OddsRatio_Clean': odds_clean,
            'Delta_P_Significance': -np.log10(p_clean) - (-np.log10(p_raw))
        })
        
    df_enrich = pd.DataFrame(enrichment_records)
    for mode in ['Raw', 'Clean']:
        sorted_p = np.sort(df_enrich[f'P_{mode}'])
        qvals = np.zeros(len(sorted_p))
        min_q = 1.0
        for i in range(len(sorted_p)-1, -1, -1):
            q = sorted_p[i] * len(sorted_p) / (i + 1)
            min_q = min(min_q, q)
            qvals[i] = min_q
        order = np.argsort(df_enrich[f'P_{mode}'])
        df_enrich[f'FDR_Q_{mode}'] = 1.0
        df_enrich.iloc[order, df_enrich.columns.get_loc(f'FDR_Q_{mode}')] = np.clip(qvals, 0.0, 1.0)
        
    df_enrich = df_enrich.sort_values(by='P_Clean', ascending=True)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df_enrich.to_csv(out_csv, index=False)
    print(f"[✓] KEGG pathway enrichment table saved to: {out_csv}")
    print("\nTop Enhanced Pathways After Error Filtering:")
    print(df_enrich[['Pathway_Description', 'Hits_Clean', 'P_Clean', 'FDR_Q_Clean', 'Hits_Raw', 'P_Raw', 'FDR_Q_Raw']].head(8).to_string(index=False))

def main():
    res_path = 'benchmark/avian_shultz2019/avian_genome_wide_filter_results.csv'
    audit_path = 'benchmark/avian_shultz2019/avian_filtered_artifacts_audit.csv'
    
    if not os.path.exists(res_path) or not os.path.exists(audit_path):
        print("[!] Results CSV files not found yet. Run after filtering completes.")
        return
        
    df = pd.read_csv(res_path)
    df_audit = pd.read_csv(audit_path)
    print(f"Loaded {len(df)} gene results and {len(df_audit)} artifact audits.")
    
    plot_heatmaps(df)
    plot_alignment_context_figure(df_audit)
    run_kegg_enrichment_comparison(df)

if __name__ == '__main__':
    main()
