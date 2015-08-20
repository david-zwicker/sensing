#!/usr/bin/env bash

sfood ../../binary_response --internal |\
    grep -v "tests.py" > dependencies.txt

sfood-graph < dependencies.txt | \
    dot -Tps | pstopdf -i

mv stdin.pdf dependencies.pdf

