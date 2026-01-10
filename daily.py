import time
import pytz

from tqdm import tqdm
from datetime import datetime
import asyncio
import subprocess


timezone = pytz.timezone("Asia/Shanghai")


def run_daily_work(cmd):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)


async def run_subprocess():
    with open("output.log", "w") as f:
        await asyncio.create_subprocess_exec(
            "python3",
            "on_recv.py",  # 命令及其参数
            stdout=f,  # 捕获标准输出
            stderr=f,  # 捕获标准错误
        )
        await asyncio.create_subprocess_exec(
            "python3",
            "stock_on_recv.py",  # 命令及其参数
            stdout=f,  # 捕获标准输出
            stderr=f,  # 捕获标准错误
        )

    # stdout, stderr = await process.communicate()  # 等待子进程完成
    # print(f"Subprocess exited with code: {process.returncode}")
    # if stdout:
    #     print(f"[STDOUT]: {stdout.decode().strip()}")
    # if stderr:
    #     print(f"[STDERR]: {stderr.decode().strip()}")


class LoadMaster:
    def __init__(self):
        self.date = "2024-12-13"

    def run(self):
        while True:
            current_date = datetime.now(timezone).date().strftime("%Y-%m-%d")
            current_time = datetime.now(timezone).time().strftime("%H:%M:%S")
            print(current_date, current_time)
            if current_date > self.date and current_time > "09:00:00":
                self.date = current_date
                asyncio.run(run_subprocess())
            time.sleep(60)


if __name__ == "__main__":
    load_master = LoadMaster()
    load_master.run()
