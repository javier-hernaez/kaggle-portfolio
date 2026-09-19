with open("06_kaggriculture/mega_simulator/evolver.py", "r", encoding="utf-8") as f:
    code = f.read()

# Replace any emoji characters that could trigger cp1252 crash
code = code.replace("🏆", "[HOT]")
code = code.replace("👑", "[NEW CHAMPION]")

# Add stdout reconfiguration at the top of evolver.py
encoding_fix = """import json
import os
import sys
import time

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
"""

code = code.replace("import json\nimport os\nimport time\n", encoding_fix)

with open("06_kaggriculture/mega_simulator/evolver.py", "w", encoding="utf-8") as f:
    f.write(code)

print("Fixed console encoding in evolver.py!")
