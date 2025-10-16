import os
# --- Configuration ---
DATA_DIR = "Stock Data Cleaned"

# --- Remove all files in the folder ---
for filename in os.listdir(DATA_DIR):
    file_path = os.path.join(DATA_DIR, filename)
    if os.path.isfile(file_path):  # ensure it's a file, not a folder
        os.remove(file_path)
        print(f"🗑️ Removed: {filename}")

print("✅ All cleaned data files removed.")
