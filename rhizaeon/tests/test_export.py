"""
tests/test_export.py
====================
Unit tests for HyPhy and NEXUS partition export bridges:
  - build_partition_intervals: 1-based codon and nucleotide coordinate framing
  - export_hyphy_partition_json: HyPhy JSON partition schema
  - export_nexus_partitions: Multi-partition NEXUS ASSUMPTIONS block
  - export_hyphy_batchfile: Standalone HyPhy batch script (.bf) generation
"""

import os
import json
import tempfile
import unittest

from rhizaeon.export import (
    build_partition_intervals,
    export_hyphy_partition_json,
    export_nexus_partitions,
    export_hyphy_batchfile
)


class TestPartitionExport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.fasta_path = os.path.join(cls.temp_dir.name, "test_align.fasta")
        with open(cls.fasta_path, "w") as f:
            f.write(">Taxon_A\n" + ("A" * 300) + "\n")
            f.write(">Taxon_B\n" + ("C" * 300) + "\n")

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def test_build_partition_intervals_codon(self):
        # 100 codons (300 nt), breakpoints at codon 30 and 70
        parts = build_partition_intervals(100, [30, 70], is_codon=True)
        self.assertEqual(len(parts), 3)

        # Partition 1: codons 0..30 -> nt 1..90
        self.assertEqual(parts[0]["start_1based"], 1)
        self.assertEqual(parts[0]["end_1based"], 90)
        self.assertEqual(parts[0]["span"], 90)

        # Partition 2: codons 30..70 -> nt 91..210
        self.assertEqual(parts[1]["start_1based"], 91)
        self.assertEqual(parts[1]["end_1based"], 210)
        self.assertEqual(parts[1]["span"], 120)

        # Partition 3: codons 70..100 -> nt 211..300
        self.assertEqual(parts[2]["start_1based"], 211)
        self.assertEqual(parts[2]["end_1based"], 300)
        self.assertEqual(parts[2]["span"], 90)

    def test_build_partition_intervals_nucleotide(self):
        # 300 nucleotides, breakpoint at nt 120
        parts = build_partition_intervals(300, [120], is_codon=False)
        self.assertEqual(len(parts), 2)
        self.assertEqual(parts[0]["start_1based"], 1)
        self.assertEqual(parts[0]["end_1based"], 120)
        self.assertEqual(parts[1]["start_1based"], 121)
        self.assertEqual(parts[1]["end_1based"], 300)

    def test_export_hyphy_partition_json(self):
        json_out = os.path.join(self.temp_dir.name, "partitions.json")
        res = export_hyphy_partition_json(100, [40], json_out, is_codon=True)

        self.assertIn("partitions", res)
        self.assertIn("partition_1", res["partitions"])
        self.assertIn("partition_2", res["partitions"])
        self.assertEqual(res["partitions"]["partition_1"]["span"], "1-120")
        self.assertEqual(res["partitions"]["partition_2"]["span"], "121-300")

        # Verify on-disk JSON file
        with open(json_out, "r") as f:
            loaded = json.load(f)
        self.assertEqual(loaded, res)

    def test_export_nexus_partitions(self):
        nex_out = os.path.join(self.temp_dir.name, "alignment.nex")
        content = export_nexus_partitions(self.fasta_path, [40], nex_out, is_codon=True)

        self.assertIn("#NEXUS", content)
        self.assertIn("BEGIN DATA;", content)
        self.assertIn("BEGIN ASSUMPTIONS;", content)
        self.assertIn("CHARSET partition_1 = 1-120;", content)
        self.assertIn("CHARSET partition_2 = 121-300;", content)
        self.assertIn("CHARPARTITION Brekpoints = partition_1:partition_1, partition_2:partition_2;", content)

        # File exists and matches
        with open(nex_out, "r") as f:
            self.assertEqual(f.read(), content)

    def test_export_hyphy_batchfile(self):
        bf_out = os.path.join(self.temp_dir.name, "screen.bf")
        content = export_hyphy_batchfile(self.fasta_path, [40], bf_out, is_codon=True)

        self.assertIn("ReadDataFile", content)
        self.assertIn("DataSetFilter filter_1 = CreateFilter (ds, 3, \"0-119\", \"\", \"\");", content)
        self.assertIn("DataSetFilter filter_2 = CreateFilter (ds, 3, \"120-299\", \"\", \"\");", content)

        with open(bf_out, "r") as f:
            self.assertEqual(f.read(), content)


if __name__ == "__main__":
    unittest.main()
