"""
Medallion Architecture - Bronze Layer: Raw Data Ingestion & Stream Partitioning.
Handles fast, zero-copy TSV scanning, schema enforcement, and country partitioning.
"""

import os
import duckdb
import pandas as pd
from typing import List, Tuple, Generator

class BronzeIngestionEngine:
    def __init__(self, threads: int = 8):
        self.con = duckdb.connect()
        self.con.execute(f"PRAGMA threads={threads}")

    def get_test_countries(self, s1_path: str) -> List[Tuple[str, int]]:
        """Detect all unique countries present in test set with record counts."""
        df = self.con.query(f"""
            SELECT country, count(*) as cnt 
            FROM read_csv('{s1_path}', delim='\\t', header=True, auto_detect=True, all_varchar=True)
            GROUP BY country ORDER BY cnt DESC
        """).df()
        return [(r['country'], int(r['cnt'])) for _, r in df.iterrows()]

    def get_all_s1_ordered(self, s1_path: str) -> List[str]:
        """Return the exact ordered list of Source 1 entity IDs."""
        return self.con.query(f"""
            SELECT entity_id 
            FROM read_csv('{s1_path}', delim='\\t', header=True, auto_detect=True, all_varchar=True)
        """).df()['entity_id'].tolist()

    def load_country_data(self, s1_path: str, s2_path: str, s3_path: str, country: str):
        """
        Load S1 and combined S2/S3 records for a specific country partition.
        Returns: (s1_df, s23_df)
        """
        s1_df = self.con.query(f"""
            SELECT entity_id, business_name, business_address, country
            FROM read_csv('{s1_path}', delim='\\t', header=True, auto_detect=True, all_varchar=True)
            WHERE country = '{country}'
        """).df()

        s23_df = self.con.query(f"""
            SELECT entity_id, business_name, business_address, country 
            FROM read_csv('{s2_path}', delim='\\t', header=True, auto_detect=True, all_varchar=True) 
            WHERE country = '{country}'
            UNION ALL
            SELECT entity_id, business_name, business_address, country 
            FROM read_csv('{s3_path}', delim='\\t', header=True, auto_detect=True, all_varchar=True) 
            WHERE country = '{country}'
        """).df()

        return s1_df, s23_df

    def load_train_sample(self, s1_path: str, s2_path: str, s3_path: str, gt_path: str, sample_size: int = 100000):
        """
        Load representative training sample with true matches and hard negative background rows.
        """
        s1_df = self.con.query(f"""
            SELECT entity_id, business_name, business_address, country
            FROM read_csv('{s1_path}', delim='\\t', header=True, auto_detect=True, all_varchar=True)
            USING SAMPLE {sample_size} ROWS
        """).df()

        gt_df = self.con.query(f"""
            SELECT source1_entity_id as s1_id, matched_entity_ids
            FROM read_csv('{gt_path}', delim='\\t', header=True, auto_detect=True, all_varchar=True)
            WHERE source1_entity_id IN (SELECT entity_id FROM s1_df)
        """).df()

        s2_df = self.con.query(f"""
            SELECT entity_id, business_name, business_address, country FROM read_csv('{s2_path}', delim='\\t', header=True, auto_detect=True, all_varchar=True)
            WHERE entity_id IN (SELECT unnest(string_split(matched_entity_ids, ',')) FROM read_csv('{gt_path}', delim='\\t', header=True, auto_detect=True, all_varchar=True) WHERE source1_entity_id IN (SELECT entity_id FROM s1_df))
            OR entity_id IN (SELECT entity_id FROM read_csv('{s2_path}', delim='\\t', header=True, auto_detect=True, all_varchar=True) USING SAMPLE 250000 ROWS)
        """).df()

        s3_df = con_q = self.con.query(f"""
            SELECT entity_id, business_name, business_address, country FROM read_csv('{s3_path}', delim='\\t', header=True, auto_detect=True, all_varchar=True)
            WHERE entity_id IN (SELECT unnest(string_split(matched_entity_ids, ',')) FROM read_csv('{gt_path}', delim='\\t', header=True, auto_detect=True, all_varchar=True) WHERE source1_entity_id IN (SELECT entity_id FROM s1_df))
            OR entity_id IN (SELECT entity_id FROM read_csv('{s3_path}', delim='\\t', header=True, auto_detect=True, all_varchar=True) USING SAMPLE 250000 ROWS)
        """).df()

        s23_pool = pd.concat([s2_df, s3_df], ignore_index=True).drop_duplicates(subset=['entity_id'])
        return s1_df, gt_df, s23_pool
