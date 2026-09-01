import os, glob, sqlite3
from collections import defaultdict
import pandas as pd
from Bio import SeqIO
from concurrent.futures import ProcessPoolExecutor

base_dir = '/Users/sergei/Projects/TOGA_MEME/benchmark/orthomam_v12'
cds_dir = os.path.join(base_dir, 'filtered_cds')
db_path = os.path.join(base_dir, 'orthomam_v12_analysis.db')

conn = sqlite3.connect(db_path)
df_masked = pd.read_sql('''
    SELECT 
        taxon,
        COUNT(*) as total_patches,
        COUNT(DISTINCT gene_id) as affected_genes,
        SUM(patch_span) as total_masked_codons,
        AVG(patch_span) as avg_patch_span
    FROM masking_audit
    GROUP BY taxon
''', conn)
conn.close()

def count_codons_in_file(fname):
    counts = defaultdict(int)
    records = list(SeqIO.parse(os.path.join(cds_dir, fname), 'fasta'))
    for r in records:
        seq = str(r.seq).upper()
        n_codons = len(seq) // 3
        valid_codons = sum(1 for i in range(n_codons) if seq[i*3:(i+1)*3] != '---')
        counts[r.id] += valid_codons
    return counts

if __name__ == '__main__':
    fasta_files = [f for f in os.listdir(cds_dir) if f.endswith('.fasta')]
    print(f'Calculating total sequenced codons across {len(fasta_files)} alignments...')

    total_sequenced = defaultdict(int)
    num_workers = max(1, os.cpu_count() - 2)
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for file_counts in executor.map(count_codons_in_file, fasta_files, chunksize=100):
            for tax, cnt in file_counts.items():
                total_sequenced[tax] += cnt

    df_seq = pd.DataFrame(list(total_sequenced.items()), columns=['taxon', 'total_sequenced_codons'])
    df_final = pd.merge(df_masked, df_seq, on='taxon', how='right').fillna({'total_patches': 0, 'affected_genes': 0, 'total_masked_codons': 0, 'avg_patch_span': 0})
    df_final['masked_fraction_pct'] = (df_final['total_masked_codons'] / df_final['total_sequenced_codons']) * 100.0
    df_final = df_final.sort_values(by='masked_fraction_pct', ascending=False)

    print('=' * 110)
    print('TOP 25 MAMMALIAN SPECIES WITH THE HIGHEST FRACTION OF MASKED SITES IN ORTHOMAM V12')
    print('=' * 110)
    print(df_final[['taxon', 'masked_fraction_pct', 'total_masked_codons', 'total_sequenced_codons', 'affected_genes', 'total_patches']].head(25).to_string(index=False))
