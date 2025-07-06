import os

# Output directory for CSV and JSON files
OUTPUT_DIR = os.path.join("outputs", "final_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Output file paths
output_csv_file = os.path.join(OUTPUT_DIR, "important_frames.csv")
output_json_file = os.path.join(OUTPUT_DIR, "results.json")
