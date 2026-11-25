"""Run as python -m scripts.acquire."""
from src.datasets import acquire, audit_all

if __name__ == "__main__":
    acquire()
    for record in audit_all():
        print(record, flush=True)
