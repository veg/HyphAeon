"""Unit tests for temporal surveillance module in hyphaeon.temporal."""
import json
import numpy as np
import pandas as pd
import pytest
import torch

from hyphaeon.temporal import (
    parse_date_to_decimal,
    extract_date_from_string,
    parse_dates_from_auspice_json,
    parse_temporal_metadata,
    infer_root_sequence,
)


class TestTemporalDateParsing:
    def test_parse_date_to_decimal_numeric(self):
        assert parse_date_to_decimal(2021.25) == 2021.25
        assert parse_date_to_decimal("2021.25") == 2021.25
        assert np.isnan(parse_date_to_decimal(1500.0))  # Out of reasonable bounds
        assert np.isnan(parse_date_to_decimal("unknown"))
        assert np.isnan(parse_date_to_decimal(None))

    def test_parse_date_to_decimal_iso(self):
        # 2021-01-01 is roughly 2021.0
        d_jan1 = parse_date_to_decimal("2021-01-01")
        assert abs(d_jan1 - 2021.0) < 0.01

        # Mid-year
        d_july = parse_date_to_decimal("2021-07-02")
        assert abs(d_july - 2021.5) < 0.02

        # Partial dates
        d_year = parse_date_to_decimal("2021")
        assert d_year == 2021.0

        # Slash formatted
        d_slash = parse_date_to_decimal("2021/07/02")
        assert abs(d_slash - 2021.5) < 0.02

    def test_extract_date_from_string(self):
        assert abs(extract_date_from_string("isolate|2021-05-15") - 2021.37) < 0.02
        assert abs(extract_date_from_string("hCoV-19/USA/123/2021-05-15") - 2021.37) < 0.02
        assert abs(extract_date_from_string("isolate/2021.45") - 2021.45) < 0.01
        assert extract_date_from_string("isolate|2021") == 2021.0
        assert np.isnan(extract_date_from_string("just_a_name_no_date"))

    def test_parse_dates_from_auspice_json(self, tmp_path):
        mock_tree = {
            "tree": {
                "name": "NODE_ROOT",
                "children": [
                    {
                        "name": "tip_1",
                        "node_attrs": {"num_date": {"value": 2021.2}}
                    },
                    {
                        "name": "tip_2",
                        "node_attrs": {"date": {"value": "2022-01-01"}}
                    },
                    {
                        "name": "internal_node",
                        "children": [
                            {
                                "name": "tip_3|2023-06-01",
                                "node_attrs": {}
                            }
                        ]
                    }
                ]
            }
        }
        json_file = tmp_path / "mock_auspice.json"
        with open(json_file, 'w') as f:
            json.dump(mock_tree, f)

        dates = parse_dates_from_auspice_json(json_file)
        assert len(dates) == 3
        assert dates["tip_1"] == 2021.2
        assert abs(dates["tip_2"] - 2022.0) < 0.01
        assert abs(dates["tip_3|2023-06-01"] - 2023.41) < 0.02

    def test_parse_temporal_metadata_table(self, tmp_path):
        csv_file = tmp_path / "meta.csv"
        df = pd.DataFrame({
            "strain": ["seq_A", "seq_B", "seq_C"],
            "collection_date": ["2020-03-15", "2021-06-20", "2022.5"]
        })
        df.to_csv(csv_file, index=False)

        dates = parse_temporal_metadata(csv_file)
        assert len(dates) == 3
        assert abs(dates["seq_A"] - 2020.20) < 0.02
        assert abs(dates["seq_B"] - 2021.46) < 0.02
        assert dates["seq_C"] == 2022.5

    def test_infer_root_sequence(self):
        # 3 codons, 4 taxa
        # site 0: A A A G -> root A
        # site 1: C C T T -> early taxa dates: 2020.0, 2020.1 -> root C
        # site 2: G G G G -> root G
        a = torch.tensor([
            [[0], [0], [0], [5]],  # site 0
            [[1], [1], [3], [3]],  # site 1
            [[2], [2], [2], [2]],  # site 2
        ], dtype=torch.long)
        taxa = ["t1", "t2", "t3", "t4"]
        taxa_dates = np.array([2020.0, 2020.1, 2022.0, 2022.5])

        root_aas, root_idx = infer_root_sequence(a, taxa, taxa_dates=taxa_dates)
        assert len(root_aas) == 3
        assert root_idx[0] == 0
        assert root_idx[1] == 1
        assert root_idx[2] == 2
