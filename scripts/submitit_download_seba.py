#!/usr/bin/env python3

import argparse
import os
import sys
import subprocess
from pathlib import Path
import download_sebea as trainer

import submitit


class Trainer(object):
    def __init__(self, download_args: list[str]):
        self.download_args = download_args

    def __call__(self) -> int:
        # Run the imported downloader's main, passing args via sys.argv
        argv_backup = list(sys.argv)
        try:
            sys.argv = ["download_sebea.py"] + self.download_args
            return trainer.main() or 0
        finally:
            sys.argv = argv_backup

    def checkpoint(self):
        return submitit.helpers.DelayedSubmission(Trainer(self.download_args))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit SEBEA dataset download job to Slurm via submitit."
    )
    # Slurm/Submitit options
    parser.add_argument("--job-name", default="sebea-download", help="Job name")
    parser.add_argument("--partition", default='cpu', help="Slurm partition/queue (e.g., gpu, cpu)")
    parser.add_argument("--qos", default=None, help="Slurm QOS")
    parser.add_argument("--account", default='beez-delta-cpu', help="Slurm account")
    parser.add_argument("--constraint", default=None, help="Slurm constraint")
    parser.add_argument("--comment", default=None, help="Slurm job comment")
    parser.add_argument("--nodes", type=int, default=1, help="Number of nodes")
    parser.add_argument("--gpus", type=int, default=0, help="Number of GPUs per task")
    parser.add_argument("--cpus", type=int, default=8, help="CPUs per task")
    parser.add_argument("--mem-gb", type=int, default=40, help="Memory (GB) per task")
    parser.add_argument("--time-min", type=int, default=1440, help="Timeout in minutes")
    parser.add_argument(
        "--log-dir",
        default=str(Path("logs") / "submitit" / "sebea"),
        help="Folder to store submitit logs",
    )
    # Everything after -- will be passed to the download script
    parser.add_argument(
        "download_args",
        nargs=argparse.REMAINDER,
        help="Arguments passed through to download_sebea.py (prefix with --)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logs_dir = Path(args.log_dir).resolve()
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Clean passthrough: argparse.REMAINDER includes a leading '--' if used
    download_args = args.download_args or []
    if download_args[:1] == ["--"]:
        download_args = download_args[1:]

    executor = submitit.AutoExecutor(folder=str(logs_dir))

    additional: dict[str, str] = {}
    if args.qos:
        additional["qos"] = args.qos
    if args.account:
        additional["account"] = args.account
    if args.constraint:
        additional["constraint"] = args.constraint
    if args.comment:
        additional["comment"] = args.comment

    gres = None
    if args.gpus and args.gpus > 0:
        gres = f"gpu:{args.gpus}"

    executor.update_parameters(
        name=args.job_name,
        timeout_min=args.time_min,
        slurm_partition=args.partition,
        cpus_per_task=args.cpus,
        mem_gb=args.mem_gb,
        nodes=args.nodes,
        slurm_gres=gres,
        slurm_additional_parameters=additional or None,
    )

    trainer_obj = Trainer(download_args)

    print(
        f"Submitting job to Slurm: name={args.job_name}, partition={args.partition}, nodes={args.nodes}, "
        f"cpus={args.cpus}, mem_gb={args.mem_gb}, gpus={args.gpus}, time_min={args.time_min}"
    )

    job = executor.submit(trainer_obj)
    print(f"Submitted job_id: {job.job_id}")
    print(f"Logs: {logs_dir}")


if __name__ == "__main__":
    main()
