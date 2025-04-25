import csv

input_path = r"C:\Users\Dylan\FPLAIFinalYearProject\external\Fantasy-Premier-League\data\2024-25\gws\merged_gw.csv"
output_path = r"C:\Users\Dylan\FPLAIFinalYearProject\data\processed\merged_gw_cleaned_consistent.csv"

# First, collect both headers
with open(input_path, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    header_42 = next(reader)
    header_49 = None
    for row in reader:
        if len(row) == 49:
            header_49 = row
            break

# Build the merged header
if header_49:
    merged_header = header_42 + header_49[42:]
else:
    merged_header = header_42

# Now process the file
with open(input_path, 'r', encoding='utf-8') as infile, \
     open(output_path, 'w', encoding='utf-8', newline='') as outfile:
    reader = csv.reader(infile)
    # Skip original header
    next(reader)
    writer = csv.writer(outfile)
    writer.writerow(merged_header)
    for row in reader:
        if len(row) == 42:
            row_out = row + [''] * (len(merged_header) - 42)
        elif len(row) == 49:
            row_out = row
        else:
            continue  # skip malformed rows
        writer.writerow(row_out)

print(f"✅ Wrote consistent file: {output_path}")
