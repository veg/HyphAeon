#!/usr/bin/env python3
"""
adaptive_simulation_engine.py

Active Learning & Gaussian Process-guided adaptive simulation framework for AxoMEME.
Efficiently maps the power landscape, identifies 50% and 80% iso-power frontiers,
and discovers sweet spots and blind spots without wasting simulation budget.
"""

import os
import sys
sys.path.insert(0, '/Users/sergei/Projects/TOGA_MEME')
import io
import re
import random
import numpy as np
import pandas as pd
import torch
import sqlite3
from Bio import Phylo, SeqIO
import pyvolve
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from scipy.stats import norm

from scripts.train_transformer_selection import PhyloAxialTransformer
from scripts.predict_regression_nexus import compute_mds_coordinates, get_codon_token, get_aa_token
from scratch.test_mg94_high_power_baseline import generate_large_diverse_tree, compute_fast_dist_matrix, compute_cf3x4_frequencies, get_all_nodes

class AdaptiveSimulationEngine:
    def __init__(self, db_path='adaptive_simulations.db', model_ckpt='axomeme_5_dim384_nonull.pt', target_power=0.50):
        self.db_path = db_path
        self.model_ckpt = model_ckpt
        self.target_power = target_power
        
        # Load AxoMEME Model
        self.device = torch.device('cpu')
        self.model = PhyloAxialTransformer(embed_dim=384, num_layers=6, num_heads=12, window_size=1).to(self.device)
        ckpt = torch.load(self.model_ckpt, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval()
        
        # Setup CF3x4 Frequencies
        pi_p1 = [0.28, 0.22, 0.32, 0.18]
        pi_p2 = [0.30, 0.24, 0.18, 0.28]
        pi_p3 = [0.15, 0.40, 0.35, 0.10]
        self.cf3x4_state_freqs = compute_cf3x4_frequencies(pi_p1, pi_p2, pi_p3)
        
        # Setup Database
        self._init_db()
        
        # GP Surrogate Model (Matérn 5/2 + Noise)
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=[1.0, 1.0, 1.0, 1.0], nu=2.5) + WhiteKernel(noise_level=0.01)
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, normalize_y=False)
        
        # Parameter Bounds: [N_taxa, T_depth, log10(beta_fg), fg_frac]
        self.param_bounds = {
            'n_taxa': (32, 384),
            't_depth': (3.0, 25.0),
            'log_beta_fg': (np.log10(2.0), np.log10(500.0)),
            'fg_frac': (0.02, 1.00)
        }
        
    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS adaptive_evaluations (
                eval_id INTEGER PRIMARY KEY AUTOINCREMENT,
                iteration INTEGER,
                n_taxa INTEGER,
                t_depth REAL,
                beta_fg REAL,
                fg_frac REAL,
                clade_mode TEXT,
                n_reps INTEGER,
                mean_power_05 REAL,
                mean_power_10 REAL,
                mean_power_20 REAL,
                mean_roc_auc REAL,
                mean_pr_lift REAL,
                mean_sel_lrt REAL,
                mean_null_lrt REAL,
                gp_predicted_mean REAL,
                gp_predicted_std REAL,
                acquisition_score REAL
            )
        ''')
        conn.commit()
        conn.close()
        
    def normalize_params(self, params_arr):
        """Maps parameter vector [N, T, log(beta), fg_frac] to [0, 1]^4."""
        n_min, n_max = self.param_bounds['n_taxa']
        t_min, t_max = self.param_bounds['t_depth']
        b_min, b_max = self.param_bounds['log_beta_fg']
        f_min, f_max = self.param_bounds['fg_frac']
        
        norm_arr = np.zeros_like(params_arr, dtype=float)
        norm_arr[:, 0] = (params_arr[:, 0] - n_min) / (n_max - n_min)
        norm_arr[:, 1] = (params_arr[:, 1] - t_min) / (t_max - t_min)
        norm_arr[:, 2] = (params_arr[:, 2] - b_min) / (b_max - b_min)
        norm_arr[:, 3] = (params_arr[:, 3] - f_min) / (f_max - f_min)
        return np.clip(norm_arr, 0.0, 1.0)
        
    def unnormalize_params(self, norm_arr):
        """Maps [0, 1]^4 back to natural parameter scales."""
        n_min, n_max = self.param_bounds['n_taxa']
        t_min, t_max = self.param_bounds['t_depth']
        b_min, b_max = self.param_bounds['log_beta_fg']
        f_min, f_max = self.param_bounds['fg_frac']
        
        raw_arr = np.zeros_like(norm_arr, dtype=float)
        raw_arr[:, 0] = norm_arr[:, 0] * (n_max - n_min) + n_min
        raw_arr[:, 1] = norm_arr[:, 1] * (t_max - t_min) + t_min
        raw_arr[:, 2] = norm_arr[:, 2] * (b_max - b_min) + b_min
        raw_arr[:, 3] = norm_arr[:, 3] * (f_max - f_min) + f_min
        return raw_arr

    def simulate_and_evaluate(self, n_taxa, t_depth, beta_fg, fg_frac, clade_mode='clade', n_reps=3):
        """Runs n_reps exact Pyvolve MG94 simulations and returns mean metrics."""
        n_taxa = int(round(n_taxa))
        t_depth = float(t_depth)
        beta_fg = float(beta_fg)
        fg_frac = float(fg_frac)
        
        powers_10 = []
        powers_05 = []
        powers_20 = []
        aucs = []
        pr_lifts = []
        sel_lrts = []
        null_lrts = []
        
        n_null = 100
        n_sel = 50
        L = n_null + n_sel
        
        for rep in range(n_reps):
            seed = random.randint(1000, 999999)
            newick_str = generate_large_diverse_tree(n_taxa=n_taxa, target_t=t_depth, seed=seed)
            py_tree = pyvolve.read_tree(tree=newick_str)
            all_nodes = get_all_nodes(py_tree)
            non_roots = [n for n in all_nodes if n.name != "root"]
            
            tree_obj = Phylo.read(io.StringIO(newick_str), 'newick')
            taxa = [term.name for term in tree_obj.get_terminals()]
            dist_mat = compute_fast_dist_matrix(tree_obj, taxa)
            mds_coords = compute_mds_coordinates(dist_mat, n_components=4)
            
            if fg_frac < 1.0:
                target_br = max(1, int(fg_frac * len(non_roots)))
                if clade_mode == 'clade':
                    internal_nodes = [n for n in all_nodes if len(n.children) > 0 and n.name != "root"]
                    clade_node = min(internal_nodes, key=lambda n: abs(len(get_all_nodes(n)) - target_br))
                    fg_nodes = set(get_all_nodes(clade_node))
                else:
                    rng = random.Random(seed)
                    fg_nodes = set(rng.sample(non_roots, target_br))
            else:
                fg_nodes = set(non_roots)
                
            def tag_nodes(n):
                if n.name != 'root':
                    n.model_flag = 'foreground' if n in fg_nodes else 'background'
                else:
                    n.model_flag = 'background'
                for ch in n.children:
                    tag_nodes(ch)
            tag_nodes(py_tree)
            
            m_null = pyvolve.Model('MG94', parameters={'alpha': 1.0, 'beta': 0.15, 'state_freqs': self.cf3x4_state_freqs})
            p_null = pyvolve.Partition(models=m_null, size=n_null)
            
            if len(fg_nodes) < len(non_roots):
                m_bg = pyvolve.Model('MG94', parameters={'alpha': 1.0, 'beta': 0.15, 'state_freqs': self.cf3x4_state_freqs}, name='background')
                m_fg = pyvolve.Model('MG94', parameters={'alpha': 1.0, 'beta': beta_fg, 'state_freqs': self.cf3x4_state_freqs}, name='foreground')
                p_sel = pyvolve.Partition(models=[m_bg, m_fg], size=n_sel, root_model_name='background')
            else:
                m_fg = pyvolve.Model('MG94', parameters={'alpha': 1.0, 'beta': beta_fg, 'state_freqs': self.cf3x4_state_freqs})
                p_sel = pyvolve.Partition(models=m_fg, size=n_sel)
                
            ev = pyvolve.Evolver(partitions=[p_null, p_sel], tree=py_tree)
            tmp_fa = f"scratch/tmp_adp_{seed}.fa"
            ev(seqfile=tmp_fa, write_anc=False, ratefile=None)
            seq_dict = {rec.id: str(rec.seq).upper() for rec in SeqIO.parse(tmp_fa, 'fasta')}
            if os.path.exists(tmp_fa):
                os.remove(tmp_fa)
                
            c_all = np.zeros((L, n_taxa, 1), dtype=np.int64)
            a_all = np.zeros((L, n_taxa, 1), dtype=np.int64)
            for i, sp in enumerate(taxa):
                seq = seq_dict[sp]
                for site in range(L):
                    codon = seq[site*3 : (site+1)*3].upper()
                    c_all[site, i, 0] = get_codon_token(codon)
                    a_all[site, i, 0] = get_aa_token(codon)
                    
            is_aa_invariable = np.zeros(L, dtype=bool)
            for site in range(L):
                aa_col = a_all[site, :, 0]
                valid_aa = aa_col[aa_col < 20]
                if len(np.unique(valid_aa)) <= 1:
                    is_aa_invariable[site] = True
                    
            c_tensor = torch.tensor(c_all, dtype=torch.long)
            a_tensor = torch.tensor(a_all, dtype=torch.long)
            d_tensor = torch.tensor(dist_mat, dtype=torch.float32).unsqueeze(0).repeat(L, 1, 1)
            z_tensor = torch.tensor(mds_coords, dtype=torch.float32).unsqueeze(0).repeat(L, 1, 1)
            
            with torch.no_grad():
                y_lrt_soft, _ = self.model(c_tensor, a_tensor, d_tensor, z_tensor)
                preds = torch.clamp(y_lrt_soft.squeeze(-1), min=0.0).numpy().flatten()
                
            preds[is_aa_invariable] = 0.0
            null_preds = preds[:n_null]
            sel_preds = preds[n_null:]
            
            y_true = np.concatenate([np.zeros(n_null), np.ones(n_sel)])
            from sklearn.metrics import roc_auc_score, average_precision_score
            auc = roc_auc_score(y_true, preds)
            pr_auc = average_precision_score(y_true, preds)
            prevalence = n_sel / L
            pr_lift = pr_auc / prevalence
            
            pwr_05 = (sel_preds >= 4.45).mean()
            pwr_10 = (sel_preds >= 3.12).mean()
            pwr_20 = (sel_preds >= 1.95).mean()
            
            powers_05.append(pwr_05)
            powers_10.append(pwr_10)
            powers_20.append(pwr_20)
            aucs.append(auc)
            pr_lifts.append(pr_lift)
            sel_lrts.append(sel_preds.mean())
            null_lrts.append(null_preds.mean())
            
        return {
            'power_05': np.mean(powers_05),
            'power_10': np.mean(powers_10),
            'power_20': np.mean(powers_20),
            'roc_auc': np.mean(aucs),
            'pr_lift': np.mean(pr_lifts),
            'sel_lrt': np.mean(sel_lrts),
            'null_lrt': np.mean(null_lrts),
        }

    def fit_gp(self):
        """Fits Gaussian Process on existing database evaluations."""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("SELECT n_taxa, t_depth, beta_fg, fg_frac, mean_power_10 FROM adaptive_evaluations", conn)
        conn.close()
        
        if len(df) < 2:
            return False
            
        X = np.column_stack([
            df['n_taxa'].values,
            df['t_depth'].values,
            np.log10(df['beta_fg'].values),
            df['fg_frac'].values
        ])
        X_norm = self.normalize_params(X)
        y = df['mean_power_10'].values
        
        self.gp.fit(X_norm, y)
        return True

    def calculate_acquisition(self, candidate_points_norm):
        """
        Straddle / Contour Uncertainty Acquisition Function:
        alpha(x) = 1.96 * sigma(x) - |mu(x) - target_power|
        """
        mu, sigma = self.gp.predict(candidate_points_norm, return_std=True)
        # Straddle score: high uncertainty + close to target power boundary
        score = 1.96 * sigma - np.abs(mu - self.target_power)
        return score, mu, sigma

    def select_next_batch(self, batch_size=4, n_candidates=1000):
        """Generates candidate points and selects top informative batch."""
        # Random candidate pool in [0, 1]^4
        cand_norm = np.random.uniform(0.0, 1.0, size=(n_candidates, 4))
        scores, mu, sigma = self.calculate_acquisition(cand_norm)
        
        # Pick top candidates with diversity penalty (greedy selection)
        selected_idx = []
        for _ in range(batch_size):
            best_idx = np.argmax(scores)
            selected_idx.append(best_idx)
            # Apply local suppression to avoid clustering
            dist = np.linalg.norm(cand_norm - cand_norm[best_idx], axis=1)
            scores[dist < 0.20] -= 10.0
            
        top_norm = cand_norm[selected_idx]
        top_raw = self.unnormalize_params(top_norm)
        
        points = []
        for i in range(batch_size):
            points.append({
                'n_taxa': int(round(top_raw[i, 0])),
                't_depth': float(top_raw[i, 1]),
                'beta_fg': float(10 ** top_raw[i, 2]),
                'fg_frac': float(top_raw[i, 3]),
                'gp_pred_mean': float(mu[selected_idx[i]]),
                'gp_pred_std': float(sigma[selected_idx[i]]),
                'acq_score': float(scores[selected_idx[i]])
            })
        return points

    def run_active_learning_loop(self, n_iterations=10, batch_size=4, initial_seed_points=6):
        print("=" * 90)
        print(f"🧠 ACTIVE LEARNING GP SIMULATION ENGINE (Target Contour: {self.target_power*100:.0f}% Power)")
        print("=" * 90)
        
        # Check existing evaluations
        conn = sqlite3.connect(self.db_path)
        count = conn.execute("SELECT COUNT(*) FROM adaptive_evaluations").fetchone()[0]
        conn.close()
        
        current_iter = 1
        if count < initial_seed_points:
            print(f"[*] Seeding engine with {initial_seed_points - count} Latin-hypercube initial points...")
            seed_norm = np.random.uniform(0.0, 1.0, size=(initial_seed_points - count, 4))
            seed_raw = self.unnormalize_params(seed_norm)
            
            for i in range(len(seed_raw)):
                n_taxa = int(round(seed_raw[i, 0]))
                t_depth = float(seed_raw[i, 1])
                beta_fg = float(10 ** seed_raw[i, 2])
                fg_frac = float(seed_raw[i, 3])
                
                print(f"  [Seed {i+1}] N={n_taxa:3d}, T={t_depth:4.1f}, beta+={beta_fg:5.1f}, fg_frac={fg_frac*100:4.1f}%")
                res = self.simulate_and_evaluate(n_taxa, t_depth, beta_fg, fg_frac, n_reps=2)
                
                conn = sqlite3.connect(self.db_path)
                conn.execute('''
                    INSERT INTO adaptive_evaluations (
                        iteration, n_taxa, t_depth, beta_fg, fg_frac, clade_mode, n_reps,
                        mean_power_05, mean_power_10, mean_power_20, mean_roc_auc, mean_pr_lift,
                        mean_sel_lrt, mean_null_lrt, gp_predicted_mean, gp_predicted_std, acquisition_score
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (0, n_taxa, t_depth, beta_fg, fg_frac, 'clade', 2,
                      res['power_05'], res['power_10'], res['power_20'], res['roc_auc'], res['pr_lift'],
                      res['sel_lrt'], res['null_lrt'], 0.5, 1.0, 0.0))
                conn.commit()
                conn.close()
                
        # Main Active Learning Iteration Loop
        for it in range(1, n_iterations + 1):
            print(f"\n🔄 --- Active Learning Iteration {it}/{n_iterations} ---")
            self.fit_gp()
            batch = self.select_next_batch(batch_size=batch_size)
            
            for b_idx, p in enumerate(batch):
                print(f"  [Query {b_idx+1}] N={p['n_taxa']:3d}, T={p['t_depth']:4.1f}, beta+={p['beta_fg']:5.1f}, fg={p['fg_frac']*100:4.1f}% | GP Pred: {p['gp_pred_mean']*100:.1f}% ± {p['gp_pred_std']*100:.1f}%")
                res = self.simulate_and_evaluate(p['n_taxa'], p['t_depth'], p['beta_fg'], p['fg_frac'], n_reps=3)
                print(f"    ↳ Observed: Power(10%)={res['power_10']*100:.1f}%, Power(20%)={res['power_20']*100:.1f}%, AUC={res['roc_auc']:.3f}, Sel LRT={res['sel_lrt']:.2f}")
                
                conn = sqlite3.connect(self.db_path)
                conn.execute('''
                    INSERT INTO adaptive_evaluations (
                        iteration, n_taxa, t_depth, beta_fg, fg_frac, clade_mode, n_reps,
                        mean_power_05, mean_power_10, mean_power_20, mean_roc_auc, mean_pr_lift,
                        mean_sel_lrt, mean_null_lrt, gp_predicted_mean, gp_predicted_std, acquisition_score
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (it, p['n_taxa'], p['t_depth'], p['beta_fg'], p['fg_frac'], 'clade', 3,
                      res['power_05'], res['power_10'], res['power_20'], res['roc_auc'], res['pr_lift'],
                      res['sel_lrt'], res['null_lrt'], p['gp_pred_mean'], p['gp_pred_std'], p['acq_score']))
                conn.commit()
                conn.close()
                
        print("\n" + "=" * 90)
        print("✅ Active Learning Battery Complete! Database updated:", self.db_path)
        print("=" * 90)

if __name__ == '__main__':
    engine = AdaptiveSimulationEngine(db_path='scratch/adaptive_simulations.db', target_power=0.50)
    engine.run_active_learning_loop(n_iterations=3, batch_size=3, initial_seed_points=4)
