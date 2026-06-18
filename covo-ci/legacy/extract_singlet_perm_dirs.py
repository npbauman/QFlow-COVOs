import os
import re
import glob

def extract_first_singlet(fci_out_path):
    """
    Parse fci.out and return the first singlet state found in the
    'Lowest FCI energies and spin multiplicities:' section.

    Returns:
        dict or None
    """
    if not os.path.exists(fci_out_path):
        return None

    with open(fci_out_path, "r") as f:
        lines = f.readlines()

    in_section = False

    # Example line:
    # state  4:  E = -7.3756009471   <S^2> = 0.00000000   S = 0.000000   multiplicity = 1
    pattern = re.compile(
        r"state\s+(\d+):\s+E\s*=\s*([-+]?\d*\.\d+|[-+]?\d+)"
        r".*?multiplicity\s*=\s*(\d+)"
    )

    for line in lines:
        if "Lowest FCI energies and spin multiplicities:" in line:
            in_section = True
            continue

        if in_section:
            match = pattern.search(line)
            if match:
                state_idx = int(match.group(1))
                energy = float(match.group(2))
                multiplicity = int(match.group(3))

                if multiplicity == 1:
                    return {
                        "state": state_idx,
                        "energy": energy,
                        "line": line.strip()
                    }

            # stop if we leave the block and hit a clearly unrelated non-state line
            elif line.strip() and not line.lstrip().startswith("state"):
                # optional: break once section is over
                continue

    return None


def main():
    perm_dirs = sorted(glob.glob("perm-*"))

    if not perm_dirs:
        print("No perm-* folders found.")
        return

    for perm_dir in perm_dirs:
        fci_out = os.path.join(perm_dir, "fci.out")
        result = extract_first_singlet(fci_out)

        if result is None:
            if not os.path.exists(fci_out):
                print(f"{perm_dir}: fci.out not found")
            else:
                print(f"{perm_dir}: no singlet state found")
        else:
            print(
                 f"{perm_dir}: first singlet -> "
                 f"state {result['state']}, E = {result['energy']:.10f}"
            )
            #print(f"    {result['line']}")


if __name__ == "__main__":
    main()
