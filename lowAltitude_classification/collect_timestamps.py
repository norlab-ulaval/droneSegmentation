from pathlib import Path
import re
from datetime import datetime


def extract_timestamp(logfile_content, message):
    pattern = re.compile(
        r"^(.*?) \[DEBUG\] - [^\s]+\.py:.*" + re.escape(message) + r".*",
        re.MULTILINE,
    )
    matches = pattern.findall(logfile_content)
    return matches if matches else []


def parse_timestamp(stamp: str) -> datetime:
    return datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S,%f")


results_path = Path("/data/droneSegResults")
experiments = [p for p in results_path.glob("*")]
all_logfiles = [sorted(exp.glob("*.txt")) for exp in experiments]
logfiles = [lf[-1] for lf in all_logfiles if len(lf) > 0]

logdir = Path("lowAltitude_classification/logs")
logdir.mkdir(exist_ok=True, parents=True)

for src_log in logfiles:
    logname = src_log.stem
    if logname.lower() == "log":
        logname = f"log_{src_log.parent.name}"
    dst_log = logdir.joinpath(f"{logname}.txt")

    if not dst_log.exists():
        dst_log.write_bytes(src_log.read_bytes())

all_logs = [p for p in logdir.glob("*.txt")]

for logpath in all_logs:
    loglines = logpath.read_text()
    start = extract_timestamp(loglines, "Fold 1/5")
    end = extract_timestamp(loglines, "Average validation accuracy across folds")
    if "filtered" in logpath.stem:
        st_time = parse_timestamp(start[-1])
        en_time = parse_timestamp(end[-1])
        delta = en_time - st_time
        print("Filtered:", delta.total_seconds(), "seconds", delta)
    # print(len(start), len(end), logpath.stem)

# pass
