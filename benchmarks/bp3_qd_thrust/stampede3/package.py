"""Assemble the standalone deployment from the scientific inputs. Refuse to replace an archive."""
import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

HERE = Path(__file__).resolve().parent
SCIENCE = HERE.parent


def merged_input():
    """Overlay the production keys on the common keys, writing each subsection once."""
    table = {}
    for name in ("bp3-common.prm", "production.prm"):
        stack = [table]
        for line in (SCIENCE / name).read_text().splitlines():
            line = line.partition("#")[0].strip()
            if line.startswith("set "):
                key, value = line[4:].split("=", 1)
                stack[-1][key.strip()] = value.strip()
            elif line.startswith("subsection "):
                stack.append(stack[-1].setdefault(line[11:], {}))
            elif line == "end":
                stack.pop()
            elif line and line != "include bp3-common.prm":
                raise ValueError(f"Unsupported input in {name}: {line}")
        if len(stack) != 1:
            raise ValueError(f"Unclosed subsection in {name}")
    table["Additional shared libraries"] = "./plugins/libbp3_qd.so"

    def lines(section, indent):
        for key, value in section.items():
            if isinstance(value, dict):
                yield indent + "subsection " + key
                yield from lines(value, indent + "  ")
                yield indent + "end"
            else:
                yield f"{indent}set {key} = {value}"
    return "# Historical 60-degree BP3-QD thrust. SI units throughout.\n" + "\n".join(lines(table, "")) + "\n"


def package(path):
    files = {"production.prm": merged_input().encode()}
    for directory, prefix, names in [
            (SCIENCE, "", ["historical-vertices.txt"]),
            (SCIENCE, "plugins/", ["CMakeLists.txt", "bp3_qd.cc", "historical_profile.cc",
                                   "event_profiles.cc", "first_event.h", "startup_cuts.cc"]),
            (SCIENCE, "tools/", ["history.py", "check_historical.py", "native_frames.py"]),
            (HERE, "", ["README.md", "submit.sh"]),
            (HERE, "tools/", ["startup.py", "run.py", "tractions.py"])]:
        for name in names:
            files[prefix + name] = (directory / name).read_bytes()
    with ZipFile(path, "x") as archive:
        for name, content in sorted(files.items()):
            entry = ZipInfo("bp3/" + name)
            entry.create_system = 3
            entry.external_attr = (0o100755 if name.endswith(".sh") else 0o100644) << 16
            archive.writestr(entry, content, compress_type=ZIP_DEFLATED)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("new_zip", type=Path)
    package(parser.parse_args().new_zip)
