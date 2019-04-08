#!/bin/bash

PROJECT_ROOT=$(dirname $(realpath "$0"))/..

cd "$PROJECT_ROOT"

#PYTHONPATH=. python examples/coseg.py --epochs 1000 --pool 0 0 0 0
#PYTHONPATH=. python examples/coseg.py --epochs 1000 --pool 0 0 0 10
#PYTHONPATH=. python examples/coseg.py --epochs 1000 --pool 5 5 5 5
PYTHONPATH=. python examples/coseg.py --epochs 1000 --pool 50 50 50 7
PYTHONPATH=. python examples/coseg.py --epochs 1000 --pool 25 25 25 3
PYTHONPATH=. python examples/coseg.py --epochs 1000 --pool 12 12 12 2




