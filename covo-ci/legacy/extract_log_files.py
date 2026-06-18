import re
import glob
import os

def extract_first_singlet(log_file):
    with open(log_file, "r") as f:
        lines = f.readlines()

    in_section = False

    pattern = re.compile(
        r"state\s+(\d+):\s+E\s*=\s*([-+]?\d*\.\d+|[-+]?\d+).*?multiplicity\s*=\s*(\d+)"
    )

    for line in lines:
        if "Lowest FCI energies and spin multiplicities:" in line:
            in_section = True
            continue

        if in_section:
            match = pattern.search(line)
            if match:
                state = int(match.group(1))
                energy = float(match.group(2))
                multiplicity = int(match.group(3))

                if multiplicity == 1:
                    return state, energy, line.strip()

    return None


# extract numeric value from filename
def perm_value(filename):
    base = os.path.basename(filename)
    match = re.search(r"perm-([0-9]+(?:\.[0-9]+)?)", base)
    return float(match.group(1)) if match else float("inf")


def main():
    log_files = glob.glob("perm*.log")

    # numeric sorting
    log_files = sorted(log_files, key=perm_value)

    for log in log_files:
        result = extract_first_singlet(log)

        if result is None:
            print(f"{log}: No singlet found")
        else:
            state, energy, line = result
            print(f"{log}: first singlet -> state {state}, E = {energy:.10f}")
            #print(f"   {line}")


if __name__ == "__main__":
    main()
