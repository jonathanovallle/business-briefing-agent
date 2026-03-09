# scripts/generate_sample_data.py
import os
os.makedirs("src/data/docs", exist_ok=True)
os.makedirs("src/data/csvs", exist_ok=True)
# generate 10 docs
for i in range(1,11):
    with open(f"src/data/docs/project_doc_{i}.txt","w",encoding="utf-8") as f:
        f.write(f"Project doc {i}\nRevenue increased by {5+i}% compared to previous period.\nMarketing spend increased in channel {i}.\nChurn changed by {(-1)**i*(i%3)}%.\nOperational notes: backlog issues: {i*2}.\n")
# generate a few CSVs
with open("src/data/csvs/financials.csv","w",encoding="utf-8") as f:
    f.write("metric,value\nrevenue,1000\ncost,400\nprofit,600\n")
with open("src/data/csvs/operations.csv","w",encoding="utf-8") as f:
    f.write("metric,value\nbacklog,12\nbugs,3\nsla_breaches,1\n")