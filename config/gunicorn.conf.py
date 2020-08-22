import multiprocessing
from os import environ

bind = ":8000"
if environ.get('PORT') is not None:
    bind = f":{environ.get('PORT')}"

workers = multiprocessing.cpu_count() * 2 + 1

