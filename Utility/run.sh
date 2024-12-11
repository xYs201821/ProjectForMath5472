#!/bin/bash

# simply run main.py 30 times
for i in {1..30}; do echo "Run #$i"; python main.py --iter $i; sleep 3; done