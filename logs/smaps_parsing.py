import re
import sys
import csv
import os

def parse_smaps_file(filepath):
    entries = []
    current = {}

    with open(filepath, 'r') as f:
        lines = f.readlines()

    for line in lines:
        # 새 메모리 블록 시작
        if re.match(r'^[0-9a-fA-F]+-[0-9a-fA-F]+', line):
            if current:
                entries.append(current)
            current = {
                "addr": line.strip(),
                "name": "(anonymous)"
            }
            parts = line.split()
            if len(parts) > 5:
                current["name"] = parts[-1]
        elif line.startswith("Size:"):
            current["size"] = line.split()[1]
        elif line.startswith("Rss:"):
            current["rss"] = line.split()[1]
        elif line.startswith("Pss:"):
            current["pss"] = line.split()[1]

    if current:
        entries.append(current)

    # 입력 파일명 기반으로 CSV 파일명 생성
    base_name = os.path.basename(filepath)
    output_csv = f"{base_name}.csv"

    # CSV 저장
    with open(output_csv, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Address Range", "Name", "Size (kB)", "RSS (kB)", "PSS (kB)"])
        for entry in entries:
            writer.writerow([
                entry["addr"],
                entry["name"],
                entry.get("size", "0"),
                entry.get("rss", "0"),
                entry.get("pss", "0")
            ])

    print(f"[✔] Parsed {len(entries)} blocks and saved to '{output_csv}'")

# 실행
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("사용법: python parse_smaps_to_csv.py <smaps_log_file>")
        sys.exit(1)
    
    smaps_file = sys.argv[1]
    parse_smaps_file(smaps_file)
