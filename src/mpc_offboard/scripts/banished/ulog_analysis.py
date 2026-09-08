#!/usr/bin/env python3
"""
ulog_analysis.py - explore the contents of a PX4 .ulg log file.

A .ulg file is a container of many "topics" (called messages). Each topic
(e.g. vehicle_local_position) is logged repeatedly over time, and every sample
has a `timestamp` field plus a bunch of data fields. pyulog reads all of that
into memory for you.

USAGE
-----
List every topic in the log (this is the place to start):
    python3 ulog_analysis.py mylog.ulg

Show the fields available inside one topic:
    python3 ulog_analysis.py mylog.ulg --topic vehicle_local_position

Print the actual numbers for some fields:
    python3 ulog_analysis.py mylog.ulg --topic vehicle_local_position --fields x y z

Plot fields against time (needs matplotlib):
    python3 ulog_analysis.py mylog.ulg --topic vehicle_local_position --fields x y z --plot

Other handy flags:
    --info       dump log metadata (duration, dropouts, system info)
    --messages   print the human-readable log/console messages
    --rows N      how many rows to print for --fields (default 10, use 0 for all)
"""

import argparse
import os
import sys

from pyulog import ULog


def load_log(path):
    """Expand ~, check the file exists, and parse it with pyulog."""
    path = os.path.expanduser(path)
    if not os.path.isfile(path):
        sys.exit(f"ERROR: file not found: {path}")
    try:
        return ULog(path)
    except Exception as exc:                       # pyulog raises plain Exceptions
        sys.exit(f"ERROR: could not parse '{path}': {exc}")


def list_topics(log):
    """Print every topic name, how many samples it has, and its field count.

    Some topics are logged on several instances (e.g. two distance sensors),
    distinguished by `multi_id`; we show that too.
    """
    print(f"{'Topic Name':<40} {'Inst':>4} {'Samples':>9} {'Fields':>7}")
    print("-" * 64)
    # Sort so the output is stable and easy to scan.
    for data in sorted(log.data_list, key=lambda d: (d.name, d.multi_id)):
        n_samples = len(data.data['timestamp'])
        print(f"{data.name:<40} {data.multi_id:>4} {n_samples:>9} {len(data.field_data):>7}")
    print(f"\n{len(log.data_list)} topic(s) total. "
          f"Use --topic <name> to inspect one.")


def show_topic(log, topic):
    """Print the fields (name + type) inside a single topic."""
    datasets = [d for d in log.data_list if d.name == topic]
    if not datasets:
        sys.exit(f"ERROR: topic '{topic}' not in log. "
                 f"Run without --topic to see available topics.")
    for data in datasets:
        n_samples = len(data.data['timestamp'])
        print(f"\nTopic: {data.name}  (instance {data.multi_id}, {n_samples} samples)")
        print(f"{'Field Name':<40} {'Type':<12}")
        print("-" * 52)
        # field_data is a list of objects with .field_name and .type_str
        for field in sorted(data.field_data, key=lambda f: f.field_name):
            print(f"{field.field_name:<40} {field.type_str:<12}")


def print_fields(log, topic, fields, rows):
    """Print the values of selected fields as a time-aligned table."""
    data = log.get_dataset(topic)                  # first instance of the topic
    available = set(data.data.keys())
    for f in fields:
        if f not in available:
            sys.exit(f"ERROR: field '{f}' not in topic '{topic}'. "
                     f"Run with --topic {topic} to list fields.")

    # timestamps are in microseconds since boot; show seconds from log start.
    t0 = data.data['timestamp'][0]
    t = (data.data['timestamp'] - t0) / 1e6
    n = len(t)
    limit = n if rows in (0, None) else min(rows, n)

    header = f"{'t[s]':>10} " + " ".join(f"{f:>14}" for f in fields)
    print(header)
    print("-" * len(header))
    for i in range(limit):
        line = f"{t[i]:>10.3f} " + " ".join(f"{data.data[f][i]:>14.5g}" for f in fields)
        print(line)
    if limit < n:
        print(f"... {n - limit} more rows (use --rows 0 to print all).")


def plot_fields(log, topic, fields):
    """Plot selected fields against time."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        sys.exit("ERROR: matplotlib not installed. Run: pip install matplotlib")

    data = log.get_dataset(topic)
    t0 = data.data['timestamp'][0]
    t = (data.data['timestamp'] - t0) / 1e6
    for f in fields:
        if f not in data.data:
            sys.exit(f"ERROR: field '{f}' not in topic '{topic}'.")
        plt.plot(t, data.data[f], label=f)
    plt.xlabel("time [s]")
    plt.title(topic)
    plt.legend()
    plt.grid(True)
    plt.show()


def show_info(log):
    """Print metadata about the log as a whole."""
    duration = (log.last_timestamp - log.start_timestamp) / 1e6
    print(f"Log duration: {duration:.1f} s")
    print(f"Topics:       {len(log.data_list)}")
    print(f"Dropouts:     {len(log.dropouts)}", end="")
    if log.dropouts:
        total = sum(d.duration for d in log.dropouts)
        print(f"  ({total} ms of data lost)")
    else:
        print()
    if log.msg_info_dict:
        print("\nSystem info:")
        for key, value in sorted(log.msg_info_dict.items()):
            print(f"  {key}: {value}")


def show_messages(log):
    """Print the console / log messages embedded in the file."""
    if not log.logged_messages:
        print("No logged messages in this file.")
        return
    for m in log.logged_messages:
        t = (m.timestamp - log.start_timestamp) / 1e6
        print(f"[{t:8.3f}s] {m.message}")


def main():
    parser = argparse.ArgumentParser(
        description="Explore the contents of a PX4 .ulg log file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("ulg", help="path to the .ulg file")
    parser.add_argument("--topic", help="inspect a single topic")
    parser.add_argument("--fields", nargs="+", help="fields to print/plot")
    parser.add_argument("--rows", type=int, default=10,
                        help="rows to print for --fields (0 = all)")
    parser.add_argument("--plot", action="store_true",
                        help="plot --fields against time")
    parser.add_argument("--info", action="store_true",
                        help="show log metadata")
    parser.add_argument("--messages", action="store_true",
                        help="show logged console messages")
    args = parser.parse_args()

    log = load_log(args.ulg)

    if args.info:
        show_info(log)
    elif args.messages:
        show_messages(log)
    elif args.topic and args.fields and args.plot:
        plot_fields(log, args.topic, args.fields)
    elif args.topic and args.fields:
        print_fields(log, args.topic, args.fields, args.rows)
    elif args.topic:
        show_topic(log, args.topic)
    else:
        list_topics(log)


if __name__ == "__main__":
    main()
