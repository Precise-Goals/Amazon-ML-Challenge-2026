import duckdb
import unicodedata
import re

print("=== INSPECTING 50 FRENCH TEST ROWS ===")
con = duckdb.connect()

q = """
SELECT entity_id, business_name, business_address, country
FROM read_csv('datasource/dataset/test/test_source1.tsv', delim='\\t', header=true)
WHERE country = 'France'
LIMIT 50;
"""
df_fr = con.query(q).df()

for i, row in enumerate(df_fr.itertuples()):
    print(f"[{i+1}] ID: {row.entity_id}")
    print(f"    Name   : {row.business_name}")
    print(f"    Address: {row.business_address}")
