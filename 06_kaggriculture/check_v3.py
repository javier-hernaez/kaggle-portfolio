import sys
sys.path.insert(0, '06_kaggriculture')

# Let's inspect src/industrial_agent.py completely to make sure we keep every good line
with open("06_kaggriculture/src/industrial_agent.py", "r") as f:
    v3_code = f.read()

print(f"Industrial v3 is {len(v3_code.splitlines())} lines long.")
