import duckdb
import json
import time
import re
import os

print("=== RUNNING PHASE A1: INVARIANT & SCRIPT AUDIT ===")
t0 = time.time()

con = duckdb.connect()

print("Setting up DuckDB views...")
con.execute("""
CREATE MACRO t(p) AS TABLE SELECT * FROM read_csv(p, delim='\\t', header=true, all_varchar=true, quote='');
CREATE VIEW gt AS SELECT * FROM t('datasource/dataset/train/train_ground_truth.tsv');
CREATE VIEW s1 AS SELECT * FROM t('datasource/dataset/train/train_source1.tsv');
CREATE VIEW pool AS SELECT * FROM t('datasource/dataset/train/train_source2.tsv')
              UNION ALL SELECT * FROM t('datasource/dataset/train/train_source3.tsv');
CREATE VIEW links AS
  SELECT source1_entity_id AS s1id, unnest(string_split(matched_entity_ids, ',')) AS cid
  FROM gt WHERE coalesce(trim(matched_entity_ids), '') <> '';
""")

# 1. I1 exclusivity
print("Evaluating I1 Exclusivity...")
df_i1 = con.execute("""
SELECT count(*) AS links, count(DISTINCT cid) AS distinct_cids, count(*) - count(DISTINCT cid) AS dup_links FROM links;
""").df()
print(df_i1.to_string(index=False))

# 2. I2 same-country rate
print("\nEvaluating I2 Same-Country Rate...")
df_i2 = con.execute("""
SELECT avg((a.country = b.country)::INT) AS same_country_rate
FROM links l 
JOIN s1 a ON l.s1id = a.entity_id 
JOIN pool b ON l.cid = b.entity_id;
""").df()
print(df_i2.to_string(index=False))

# 3. I3 per-source cardinality
print("\nEvaluating I3 Per-Source Cardinality...")
df_i3 = con.execute("""
SELECT left(cid, 2) AS src, n, count(*) AS pair_count, count(DISTINCT s1id) AS s1_count
FROM (
  SELECT s1id, cid, count(*) OVER (PARTITION BY s1id, left(cid, 2)) AS n FROM links
)
GROUP BY src, n
ORDER BY src, n;
""").df()
print(df_i3.to_string(index=False))

# 4. I5 distractor rate
print("\nEvaluating I5 Distractor Rate...")
df_i5 = con.execute("""
SELECT 1 - (SELECT count(DISTINCT cid) FROM links)::DOUBLE / (SELECT count(*) FROM pool) AS distractor_rate;
""").df()
print(df_i5.to_string(index=False))

# 5. Non-Latin scripts per country
print("\nEvaluating Script Counts (Devanagari vs Non-ASCII)...")
# Check on s1
df_scripts_s1 = con.execute("""
SELECT 
    country,
    count(*) AS total_records,
    sum(CASE WHEN regexp_matches(business_name || ' ' || business_address, '[\\x{0900}-\\x{097F}]') THEN 1 ELSE 0 END) AS devanagari_count,
    round(100.0 * sum(CASE WHEN regexp_matches(business_name || ' ' || business_address, '[\\x{0900}-\\x{097F}]') THEN 1 ELSE 0 END) / count(*), 4) AS devanagari_pct,
    sum(CASE WHEN regexp_matches(business_name || ' ' || business_address, '[^\\x00-\\x7F]') THEN 1 ELSE 0 END) AS non_ascii_count,
    round(100.0 * sum(CASE WHEN regexp_matches(business_name || ' ' || business_address, '[^\\x00-\\x7F]') THEN 1 ELSE 0 END) / count(*), 4) AS non_ascii_pct
FROM s1
GROUP BY country;
""").df()
print("Train Source 1 Scripts:")
print(df_scripts_s1.to_string(index=False))

# Check on pool
df_scripts_pool = con.execute("""
SELECT 
    country,
    count(*) AS total_records,
    sum(CASE WHEN regexp_matches(business_name || ' ' || business_address, '[\\x{0900}-\\x{097F}]') THEN 1 ELSE 0 END) AS devanagari_count,
    round(100.0 * sum(CASE WHEN regexp_matches(business_name || ' ' || business_address, '[\\x{0900}-\\x{097F}]') THEN 1 ELSE 0 END) / count(*), 4) AS devanagari_pct,
    sum(CASE WHEN regexp_matches(business_name || ' ' || business_address, '[^\\x00-\\x7F]') THEN 1 ELSE 0 END) AS non_ascii_count,
    round(100.0 * sum(CASE WHEN regexp_matches(business_name || ' ' || business_address, '[^\\x00-\\x7F]') THEN 1 ELSE 0 END) / count(*), 4) AS non_ascii_pct
FROM pool
GROUP BY country;
""").df()
print("\nTrain Pool (S2 + S3) Scripts:")
print(df_scripts_pool.to_string(index=False))

# Check on test sets as well
df_scripts_test = con.execute("""
SELECT 
    country,
    count(*) AS total_records,
    sum(CASE WHEN regexp_matches(business_name || ' ' || business_address, '[\\x{0900}-\\x{097F}]') THEN 1 ELSE 0 END) AS devanagari_count,
    round(100.0 * sum(CASE WHEN regexp_matches(business_name || ' ' || business_address, '[\\x{0900}-\\x{097F}]') THEN 1 ELSE 0 END) / count(*), 4) AS devanagari_pct,
    sum(CASE WHEN regexp_matches(business_name || ' ' || business_address, '[^\\x00-\\x7F]') THEN 1 ELSE 0 END) AS non_ascii_count,
    round(100.0 * sum(CASE WHEN regexp_matches(business_name || ' ' || business_address, '[^\\x00-\\x7F]') THEN 1 ELSE 0 END) / count(*), 4) AS non_ascii_pct
FROM t('datasource/dataset/test/test_source1.tsv')
GROUP BY country;
""").df()
print("\nTest Source 1 Scripts:")
print(df_scripts_test.to_string(index=False))

duration = time.time() - t0
print(f"\nExecution finished in {duration:.1f}s")

# Log to reports/experiments.jsonl
experiment_log = {
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "phase": "A1_invariants",
    "i1_links": int(df_i1['links'][0]),
    "i1_distinct_cids": int(df_i1['distinct_cids'][0]),
    "i1_dup_links": int(df_i1['dup_links'][0]),
    "i2_same_country_rate": float(df_i2['same_country_rate'][0]),
    "i5_distractor_rate": float(df_i5['distractor_rate'][0]),
    "duration_sec": round(duration, 2)
}

os.makedirs("reports", exist_ok=True)
with open("reports/experiments.jsonl", "a", encoding="utf-8") as f:
    f.write(json.dumps(experiment_log) + "\n")

print("Result logged to reports/experiments.jsonl")
