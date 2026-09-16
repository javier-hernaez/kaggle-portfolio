with open("06_kaggriculture/main.py", "r", encoding="utf-8") as f:
    code = f.read()

# Replace SHED_ADJACENT
code = code.replace(
    "SHED_ADJACENT = {(4, 4), (5, 4), (4, 5), (5, 5), (3, 4), (4, 3)}",
    "SHED_ADJACENT = {(4, 4), (5, 4), (4, 5), (5, 5)}"
)

with open("06_kaggriculture/main.py", "w", encoding="utf-8") as f:
    f.write(code)

print("Fixed SHED_ADJACENT!")
