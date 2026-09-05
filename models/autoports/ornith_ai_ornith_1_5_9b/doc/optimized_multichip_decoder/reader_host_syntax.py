import json
import shlex
import subprocess
from pathlib import Path

root = Path("/tmp/ornith-reader-mesh-autofix")
rows = json.loads((root / "reader_compile_commands.json").read_text())
sources = [
    "device/utilities/matmul_utilities.cpp",
    "device/factory/matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp",
    "device/matmul_device_operation.cpp",
    "matmul_nanobind.cpp",
]
results = []
for source in sources:
    row = next(
        row
        for row in rows
        if Path(row["file"]).exists() and "/operations/matmul/" + source in Path(row["file"]).read_text()
    )
    original = shlex.split(row["command"])
    original = original[original.index("/usr/bin/clang++-20") :]
    args = []
    index = 0
    while index < len(original):
        arg = original[index]
        if arg in ["-o", "-MF", "-MT", "-c"]:
            index += 2
            continue
        if arg == "-MD":
            index += 1
            continue
        if arg == "-Xclang" and original[index + 1] in ["-include-pch", "-include"]:
            index += 4
            continue
        if arg.startswith("-I/work"):
            suffix = arg[len("-I/work") :]
            if not suffix.startswith(("/build_Release", "/.cpmcache")):
                args.append("-I" + str(root) + suffix)
        args.append(arg)
        index += 1
    args += [
        "-fsyntax-only",
        "-DCMAKE_UNIQUE_NAMESPACE=reader_host_syntax",
        str(root / "ttnn/cpp/ttnn/operations/matmul" / source),
    ]
    log = root / ("reader_syntax_" + Path(source).stem + ".log")
    result = subprocess.run(args, cwd=root, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log.write_text(result.stdout)
    results.append(dict(source=source, command=args, log=str(log), exit_code=result.returncode))
    (root / "reader_syntax_results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(source, "exit", result.returncode, "log", log, flush=True)
    print(result.stdout[-3000:], flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)
