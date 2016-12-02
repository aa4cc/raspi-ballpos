#!/usr/bin/python3


import click
import hooppos
import time

@click.command()
@click.option('--count', '-c', default=1, help="Count of measured samles")
@click.option('--pause', '-p', default=0.1, help="Pause between samples")
def main(count, pause):
	for _ in range(count):
		print("{} {}".format(*hooppos.measpos_read()))
		time.sleep(pause)


main()