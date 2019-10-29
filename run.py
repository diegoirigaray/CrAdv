import sys


if __name__ == "__main__":
    from src.exec.scheduler import Scheduler

    file_name = sys.argv[1] if len(sys.argv) > 1 else None
    if not file_name:
        raise Exception("No tasks file specified.")
    scheduler = Scheduler()
    scheduler(file_name)
