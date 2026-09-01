import os, sys, time, glob, sqlite3
sys.path.insert(0, '/Users/sergei/Projects/TOGA_MEME/axomeme_repo')

import torch
import numpy as np
import pandas as pd
from scipy import stats
from hyphaeon.model import PhyloAxialTransformer
from hyphaeon.weights import load_arch_config, load_weights
from hyphaeon.dataset import load_alignment_and_tree

base_dir = '/Users/sergei/Projects/TOGA_MEME/benchmark/orthomam_v12'
masked_cds_dir = os.path.join(base_dir, 'masked_cds')
trees_dir = os.path.join(base_dir, 'trees')
db_path = os.path.join(base_dir, 'orthomam_v12_analysis.db')
weights_file = '/Users/sergei/Projects/TOGA_MEME/axomeme_repo/model.safetensors'

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f'Starting OrthoMaM selection inference on device: {device}')

# Initialize SQLite database tables
conn = sqlite3.connect(db_path)
conn.execute('''
CREATE TABLE IF NOT EXISTS orthomam_gene_selection (
    gene_id INTEGER PRIMARY KEY,
    gene_symbol TEXT,
    num_taxa INTEGER,
    num_codons INTEGER,
    cct_p_value REAL,
    sig_sites_p05 INTEGER,
    sig_sites_q10 INTEGER,
    mean_lrt REAL,
    max_lrt REAL,
    inference_time_sec REAL
);
''')
conn.execute('''
CREATE TABLE IF NOT EXISTS orthomam_site_selection (
    gene_id INTEGER,
    site_1idx INTEGER,
    lrt REAL,
    p_value REAL,
    q_value REAL,
    is_sig_p05 INTEGER,
    is_sig_q10 INTEGER,
    PRIMARY KEY (gene_id, site_1idx)
);
''')
conn.execute('CREATE INDEX IF NOT EXISTS idx_gene_cct ON orthomam_gene_selection(cct_p_value);')
conn.execute('CREATE INDEX IF NOT EXISTS idx_site_q ON orthomam_site_selection(q_value);')
conn.commit()

# Check existing processed genes
cur = conn.cursor()
cur.execute('SELECT gene_id FROM orthomam_gene_selection')
already_done = set(row[0] for row in cur.fetchall())
print(f'Found {len(already_done)} genes already evaluated in database.')

# Load model
config = load_arch_config(weights=weights_file)
model = PhyloAxialTransformer(
    embed_dim=config['embed_dim'],
    num_layers=config['num_layers'],
    num_heads=config['num_heads'],
    window_size=config['window_size']
).to(device)
state_dict = load_weights(weights=weights_file, map_location=device)
model.load_state_dict(state_dict, strict=False)
model.eval()

# Load gene symbol mapping
sym_map = {}
if os.path.exists(os.path.join(base_dir, 'orthomam_v12_masked_artifacts_audit.csv')):
    df_audit = pd.read_csv(os.path.join(base_dir, 'orthomam_v12_masked_artifacts_audit.csv'), usecols=['Gene_ID', 'Gene_Symbol']).drop_duplicates()
    sym_map = dict(zip(df_audit['Gene_ID'], df_audit['Gene_Symbol']))

# List all masked fasta files
fasta_files = sorted(os.listdir(masked_cds_dir))
to_process = []
for f in fasta_files:
    if not f.endswith('.fasta'): continue
    g_id = int(f.split('_')[0])
    if g_id not in already_done:
        to_process.append((g_id, f))

print(f'Total genes to evaluate: {len(to_process)}')

def calc_cct(pvals):
    pvals = np.clip(pvals, 1e-15, 1.0 - 1e-15)
    t_cct = np.mean(np.tan((0.5 - pvals) * np.pi))
    p_cct = 0.5 - (np.arctan(t_cct) / np.pi)
    return float(np.clip(p_cct, 0.0, 1.0))

batch_gene_records = []
batch_site_records = []
t_total_start = time.time()
processed_count = 0

for g_id, f in to_process:
    t_gene_start = time.time()
    aln_path = os.path.join(masked_cds_dir, f)
    tree_path = os.path.join(trees_dir, f.replace('.fasta', '.rootree'))
    if not os.path.exists(tree_path):
        tree_path = None
        
    try:
        c_tensor, a_tensor, d_mat, z_coords, inv_mask, taxa, L = load_alignment_and_tree(
            aln_path, tree_path, prune_duplicates=True
        )
    except Exception as e:
        print(f'Error loading gene {g_id}: {e}')
        continue
        
    N = len(taxa)
    var_indices = np.where(~inv_mask)[0]
    n_var = len(var_indices)
    
    lrts = np.zeros(L, dtype=np.float32)
    
    if n_var > 0:
        eff_batch_size = 64 if N < 300 else 32
        tree_cache = model.precompute_tree_cache(d_mat.to(device), z_coords.to(device))
        
        with torch.no_grad():
            for b_start in range(0, n_var, eff_batch_size):
                b_end = min(b_start + eff_batch_size, n_var)
                batch_idx = var_indices[b_start:b_end]
                c_ch = c_tensor[batch_idx].to(device)
                a_ch = a_tensor[batch_idx].to(device)
                y_soft, _ = model.forward_cached(c_ch, a_ch, tree_cache)
                lrts[batch_idx] = torch.clamp(y_soft.squeeze(-1), min=0.0).cpu().numpy().flatten()
                del c_ch, a_ch, y_soft
        del tree_cache, d_mat, z_coords
        if device.type == 'mps':
            torch.mps.empty_cache()
            
    # Compute mixture asymptotic p-values
    pvals = np.full(L, 2.0 / 3.0, dtype=np.float32)
    pos_mask = lrts > 0.0
    if np.any(pos_mask):
        pvals[pos_mask] = (2.0 / 3.0) * (0.45 * stats.chi2.sf(lrts[pos_mask], df=1) + 0.55 * stats.chi2.sf(lrts[pos_mask], df=2))
        
    # FDR q-values (Benjamini-Hochberg)
    order = np.argsort(pvals)
    ranks = np.empty(L, dtype=int)
    ranks[order] = np.arange(1, L + 1)
    raw_q = pvals * (L / ranks)
    sorted_q = raw_q[order]
    for i in range(L - 2, -1, -1):
        sorted_q[i] = min(sorted_q[i], sorted_q[i + 1])
    raw_q[order] = sorted_q
    qvals = np.clip(raw_q, 0.0, 1.0).astype(np.float32)
    
    cct_p = calc_cct(pvals)
    sig_p05 = int((pvals <= 0.05).sum())
    sig_q10 = int((qvals <= 0.10).sum())
    mean_lrt = float(np.mean(lrts))
    max_lrt = float(np.max(lrts))
    elapsed_gene = time.time() - t_gene_start
    
    g_sym = sym_map.get(g_id, f'GENE_{g_id}')
    batch_gene_records.append((
        g_id, g_sym, N, L, cct_p, sig_p05, sig_q10, round(mean_lrt, 4), round(max_lrt, 4), round(elapsed_gene, 4)
    ))
    
    # Store sites with p <= 0.20 or variable sites
    for s_idx in range(L):
        if pvals[s_idx] <= 0.20 or qvals[s_idx] <= 0.20:
            batch_site_records.append((
                g_id, s_idx + 1, round(float(lrts[s_idx]), 4), round(float(pvals[s_idx]), 6),
                round(float(qvals[s_idx]), 6), int(pvals[s_idx] <= 0.05), int(qvals[s_idx] <= 0.10)
            ))
            
    processed_count += 1
    
    # Flush to SQLite every 50 genes
    if len(batch_gene_records) >= 50:
        cur.executemany('''
            INSERT OR REPLACE INTO orthomam_gene_selection 
            (gene_id, gene_symbol, num_taxa, num_codons, cct_p_value, sig_sites_p05, sig_sites_q10, mean_lrt, max_lrt, inference_time_sec)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', batch_gene_records)
        if batch_site_records:
            cur.executemany('''
                INSERT OR REPLACE INTO orthomam_site_selection 
                (gene_id, site_1idx, lrt, p_value, q_value, is_sig_p05, is_sig_q10)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', batch_site_records)
        conn.commit()
        batch_gene_records = []
        batch_site_records = []
        
        rate = processed_count / (time.time() - t_total_start)
        eta_sec = (len(to_process) - processed_count) / max(1e-3, rate)
        print(f'[Progress] Processed {processed_count}/{len(to_process)} genes ({processed_count/len(to_process)*100:.1f}%) | {rate:.2f} genes/sec | ETA: {eta_sec/60:.1f} min')

if batch_gene_records:
    cur.executemany('''
        INSERT OR REPLACE INTO orthomam_gene_selection 
        (gene_id, gene_symbol, num_taxa, num_codons, cct_p_value, sig_sites_p05, sig_sites_q10, mean_lrt, max_lrt, inference_time_sec)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', batch_gene_records)
    if batch_site_records:
        cur.executemany('''
            INSERT OR REPLACE INTO orthomam_site_selection 
            (gene_id, site_1idx, lrt, p_value, q_value, is_sig_p05, is_sig_q10)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', batch_site_records)
    conn.commit()

conn.close()
print(f'🎉 Completed whole-exome selection inference across {processed_count} OrthoMaM genes in {(time.time()-t_total_start)/60:.2f} minutes!')
