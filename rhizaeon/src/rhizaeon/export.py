"""
rhizaeon.export
===============
HyPhy and NEXUS partition export bridges for downstream selection analysis
(GARD, BUSTED, MEME, FEL, aBSREL).
"""

from typing import List, Dict, Optional
from pathlib import Path
import json


def build_partition_intervals(
    num_units: int,
    breakpoints: List[int],
    is_codon: bool = True
) -> List[Dict[str, int]]:
    """
    Constructs 1-indexed start and end coordinates for partitions separated
    by breakpoints.
    
    If is_codon is True, unit positions are multiplied by 3 to give nucleotide coordinates.
    """
    sorted_bps = sorted(list(set(breakpoints)))
    # Ensure breakpoints are valid internal cut points
    sorted_bps = [b for b in sorted_bps if 0 < b < num_units]

    boundaries = [0] + sorted_bps + [num_units]
    partitions = []

    for idx in range(len(boundaries) - 1):
        u_start = boundaries[idx]
        u_end = boundaries[idx + 1]

        if is_codon:
            nt_start = u_start * 3 + 1
            nt_end = u_end * 3
        else:
            nt_start = u_start + 1
            nt_end = u_end

        partitions.append({
            "name": f"partition_{idx + 1}",
            "unit_start": u_start,
            "unit_end": u_end,
            "start_1based": nt_start,
            "end_1based": nt_end,
            "span": nt_end - nt_start + 1
        })

    return partitions


def export_hyphy_partition_json(
    num_units: int,
    breakpoints: List[int],
    output_path: str,
    is_codon: bool = True
) -> Dict:
    """
    Exports partition definitions into standard HyPhy partition JSON format,
    suitable for `hyphy busted --alignment ... --partitions ...`.
    """
    partitions = build_partition_intervals(num_units, breakpoints, is_codon=is_codon)

    hyphy_dict = {
        "partitions": {}
    }

    for p in partitions:
        hyphy_dict["partitions"][p["name"]] = {
            "span": f"{p['start_1based']}-{p['end_1based']}",
            "filter": f"{p['start_1based']}-{p['end_1based']}"
        }

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(hyphy_dict, f, indent=2)

    return hyphy_dict


def export_nexus_partitions(
    fasta_path: str,
    breakpoints: List[int],
    output_path: str,
    is_codon: bool = True
) -> str:
    """
    Reads a FASTA alignment and writes a multi-partition NEXUS file with
    CHARSET and CHARPARTITION declarations in the ASSUMPTIONS block.
    """
    from rhizaeon.tensor import parse_fasta

    taxa, seqs = parse_fasta(fasta_path)
    num_taxa = len(taxa)
    num_nt = len(seqs[0])
    num_units = num_nt // 3 if is_codon else num_nt

    partitions = build_partition_intervals(num_units, breakpoints, is_codon=is_codon)

    lines = [
        "#NEXUS",
        "BEGIN DATA;",
        f"    DIMENSIONS NTAX={num_taxa} NCHAR={num_nt};",
        "    FORMAT DATATYPE=DNA GAP=- MISSING=? MATCHCHAR=.;",
        "    MATRIX"
    ]

    max_name_len = max(len(t) for t in taxa)
    for t, s in zip(taxa, seqs):
        lines.append(f"    {t:<{max_name_len + 4}}{s}")

    lines.append("    ;")
    lines.append("END;")
    lines.append("")
    lines.append("BEGIN ASSUMPTIONS;")

    charset_names = []
    for p in partitions:
        c_name = p["name"]
        charset_names.append(c_name)
        lines.append(f"    CHARSET {c_name} = {p['start_1based']}-{p['end_1based']};")

    part_spec = ", ".join(f"{name}:{name}" for name in charset_names)
    lines.append(f"    CHARPARTITION Breakpoints = {part_spec};")
    lines.append("END;")

    content = "\n".join(lines) + "\n"

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        f.write(content)

    return content


def export_split_fastas(
    fasta_path: str,
    breakpoints: List[int],
    output_dir: str,
    is_codon: bool = False
) -> List[str]:
    """
    Slices the alignment into disjoint non-recombinant FASTA files,
    one for each partition: partition_1.fasta, partition_2.fasta, ...
    
    This disassembles the dataset into non-recombinant components for
    independent downstream phylogenetic tree estimation (IQ-TREE, RAxML, FastTree).
    """
    from rhizaeon.tensor import parse_fasta

    taxa, seqs = parse_fasta(fasta_path)
    num_nt = len(seqs[0])
    num_units = num_nt // 3 if is_codon else num_nt

    partitions = build_partition_intervals(num_units, breakpoints, is_codon=is_codon)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    created_paths = []
    for p in partitions:
        part_name = p["name"]
        start_0 = p["start_1based"] - 1
        end_0 = p["end_1based"]
        
        file_path = out_dir / f"{part_name}.fasta"
        with open(file_path, "w", encoding="utf-8") as f:
            for t, s in zip(taxa, seqs):
                sub_seq = s[start_0:end_0]
                f.write(f">{t}\n{sub_seq}\n")
        created_paths.append(str(file_path))

    return created_paths


def export_hyphy_batchfile(
    fasta_path: str,
    breakpoints: List[int],
    output_path: str,
    is_codon: bool = True
) -> str:
    """
    Generates a standalone HyPhy batchfile (.bf) that initializes data filters
    for each recombinant partition.
    """
    from rhizaeon.tensor import parse_fasta

    taxa, seqs = parse_fasta(fasta_path)
    num_nt = len(seqs[0])
    num_units = num_nt // 3 if is_codon else num_nt

    partitions = build_partition_intervals(num_units, breakpoints, is_codon=is_codon)

    lines = [
        "/* ------------------------------------------------------------- */",
        "/* RhizAeon Multi-Partition Batchfile for HyPhy                  */",
        "/* ------------------------------------------------------------- */",
        f'DataSet ds = ReadDataFile ("{Path(fasta_path).resolve()}");',
        ""
    ]

    for idx, p in enumerate(partitions, 1):
        if is_codon:
            # HyPhy codon filters specify step 3
            # 0-based coordinate string: "start-end"
            start_0 = p["unit_start"] * 3
            end_0 = p["unit_end"] * 3 - 1
            lines.append(f'DataSetFilter filter_{idx} = CreateFilter (ds, 3, "{start_0}-{end_0}", "", "");')
        else:
            start_0 = p["start_1based"] - 1
            end_0 = p["end_1based"] - 1
            lines.append(f'DataSetFilter filter_{idx} = CreateFilter (ds, 1, "{start_0}-{end_0}", "", "");')

    lines.append("")
    lines.append('fprintf (stdout, "[✓] Initialized " + ' + str(len(partitions)) + ' + " recombinant partitions from RhizAeon.\\n");')

    content = "\n".join(lines) + "\n"

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        f.write(content)

    return content
