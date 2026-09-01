import os, time, sqlite3
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

base_dir = '/Users/sergei/Projects/TOGA_MEME/benchmark/orthomam_v12'
in_cds_dir = os.path.join(base_dir, 'filtered_cds')
out_cds_dir = os.path.join(base_dir, 'masked_cds')
db_path = os.path.join(base_dir, 'orthomam_v12_analysis.db')

os.makedirs(out_cds_dir, exist_ok=True)

# 1. Load masking map from db
conn = sqlite3.connect(db_path)
cur = conn.cursor()
cur.execute('SELECT gene_id, taxon, patch_start, patch_end FROM masking_audit')
rows = cur.fetchall()
conn.close()

gene_masks = defaultdict(lambda: defaultdict(list))
for g_id, taxon, p_s, p_e in rows:
    gene_masks[g_id][taxon].append((p_s, p_e))

print(f'Loaded masking coordinates for {len(gene_masks)} genes.')

def process_file(fasta_file):
    g_id_str = fasta_file.split('_')[0]
    try:
        g_id = int(g_id_str)
    except ValueError:
        g_id = None
        
    in_path = os.path.join(in_cds_dir, fasta_file)
    out_path = os.path.join(out_cds_dir, fasta_file)
    
    if g_id not in gene_masks:
        records = list(SeqIO.parse(in_path, 'fasta'))
        SeqIO.write(records, out_path, 'fasta')
        return (fasta_file, 0)
        
    masks_by_taxon = gene_masks[g_id]
    records = list(SeqIO.parse(in_path, 'fasta'))
    masked_records = []
    total_masked_patches = 0
    
    for r in records:
        s = str(r.seq)
        if r.id in masks_by_taxon:
            s_list = list(s)
            for p_s, p_e in masks_by_taxon[r.id]:
                nt_start = (p_s - 1) * 3
                nt_end = p_e * 3
                for idx in range(nt_start, min(nt_end, len(s_list))):
                    s_list[idx] = 'N'
                total_masked_patches += 1
            s = ''.join(s_list)
            
        masked_records.append(SeqRecord(Seq(s), id=r.id, description=''))
        
    SeqIO.write(masked_records, out_path, 'fasta')
    return (fasta_file, total_masked_patches)

if __name__ == '__main__':
    fasta_files = [f for f in os.listdir(in_cds_dir) if f.endswith('.fasta')]
    print(f'Processing {len(fasta_files)} alignments across CPUs...')

    t0 = time.time()
    num_workers = max(1, os.cpu_count() - 2)
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(process_file, fasta_files, chunksize=50))

    elapsed = time.time() - t0
    total_patches_applied = sum(r[1] for r in results)
    print(f'Done generating {len(results)} masked alignments in {elapsed:.2f}s!')
    print(f'Total mask patches applied: {total_patches_applied:,}')
