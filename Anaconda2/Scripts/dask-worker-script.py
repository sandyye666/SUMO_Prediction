if __name__ == '__main__':
    import sys
    import distributed.cli.dask_worker

    sys.exit(distributed.cli.dask_worker.go())
