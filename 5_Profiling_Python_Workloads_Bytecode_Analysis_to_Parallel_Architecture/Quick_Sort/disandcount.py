import marshal
import dis
import collections

def count_instructions(code_obj, counter):
    """
    Recursively counts the opcodes in a code object and its nested code objects.
    """
    for instr in dis.get_instructions(code_obj):
        counter[instr.opname] += 1
    for const in code_obj.co_consts:
        if isinstance(const, type(code_obj)):
            count_instructions(const, counter)

# Path to the compiled bytecode file (generated using py_compile)
pyc_file = r"C:\Users\jaswa\Challenge5\Quick_Sort\__pycache__\quick_sort.cpython-312.pyc"
output_file = "instruction_counts.txt"
header_size = 16  # Adjust header size if necessary for your Python version

# Load the code object from the .pyc file
with open(pyc_file, "rb") as f:
    f.read(header_size)  # Skip the header
    code_obj = marshal.load(f)

# Count instructions recursively in the code object and its nested code objects
counter = collections.Counter()
count_instructions(code_obj, counter)

# Write the instruction counts into an output file
with open(output_file, "w") as f:
    for opname, count in sorted(counter.items()):
        f.write(f"{opname}: {count}\n")

print(f"Instruction counts written to {output_file}")