#!/bin/bash

PROJECT_ROOT=$(dirname $(realpath "$0"))/..

cd "$PROJECT_ROOT"

PYTHONPATH=. python examples/coseg.py --epochs $1


