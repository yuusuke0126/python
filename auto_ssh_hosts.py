#!/usr/bin/env python3
"""
auto_ssh_hosts.py

使い方例:
  ./auto_ssh_hosts.py --user youruser host1 host2
  ./auto_ssh_hosts.py --user youruser --hosts-file hosts.txt

hosts は /etc/hosts に書かれた名前でも、IP アドレスでも可。
"""

import argparse
import subprocess
import time
import shutil
import sys

PING_INTERVAL = 2  # 秒

def ping_once(target, timeout_sec=2):
    """ping -c 1 を実行して応答があれば True を返す。"""
    # -W は応答タイムアウト（秒）(Linux ping)
    cmd = ["ping", "-c", "1", "-W", str(timeout_sec), target]
    try:
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return res.returncode == 0
    except FileNotFoundError:
        print("ping コマンドが見つかりません。", file=sys.stderr)
        return False

def open_terminal_and_ssh(user, host):
    """gnome-terminal を開いて ssh user@host を実行する。"""
    term = shutil.which("gnome-terminal")
    if not term:
        print("gnome-terminal が見つかりません。端末を手動で開いて ssh してください。", file=sys.stderr)
        return
    # -- は以降をコマンドとして扱う（gnome-terminal の挙動に依存）
    subprocess.Popen([term, "--", "ssh", f"{user}@{host}"])

def monitor_hosts(hosts, user, interval=PING_INTERVAL):
    """hosts（リスト）をループ監視。最初に応答が返ってきたら SSH を開いて終了。"""
    print("Monitoring:", ", ".join(hosts))
    try:
        while True:
            for h in hosts:
                print(f"checking {h}...", end="", flush=True)
                if ping_once(h):
                    print(" reachable!")
                    open_terminal_and_ssh(user, h)
                    return
                else:
                    print(" no response")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting.")

def read_hosts_file(path):
    """1行に1ホストのテキストファイルを読み込む（コメント・空行を無視）"""
    out = []
    with open(path, "r") as f:
        for line in f:
            s = line.split("#", 1)[0].strip()
            if s:
                out.append(s)
    return out

def main():
    p = argparse.ArgumentParser(description="Wait until a host responds, then open terminal and ssh.")
    p.add_argument("hosts", nargs="*", help="hostname or IP (can be names from /etc/hosts)")
    p.add_argument("--hosts-file", "-f", help="file with one host per line")
    p.add_argument("--user", "-u", required=True, help="ssh username")
    p.add_argument("--interval", "-i", type=float, default=PING_INTERVAL, help="poll interval seconds")
    args = p.parse_args()

    hosts = list(args.hosts)
    if args.hosts_file:
        hosts += read_hosts_file(args.hosts_file)

    if not hosts:
        print("監視するホストを指定してください（引数または --hosts-file）。", file=sys.stderr)
        sys.exit(1)

    monitor_hosts(hosts, args.user, args.interval)

if __name__ == "__main__":
    main()
