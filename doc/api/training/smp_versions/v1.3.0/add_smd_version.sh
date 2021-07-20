#!/usr/bin/env python
# add_no_index2.py
import fileinput
import sys

for line in fileinput.input(inplace=True):
    if '.. class::' in line or '.. function::' in line or '.. data::' in line or '.. _' in line:
        sys.stdout.write(line + '   :noindex:\n')
    else:
        sys.stdout.write(line)
