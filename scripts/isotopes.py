#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script accepts one input file (e.g. input.py) with the RMG-Py model to generate, 
optional parameters `--original [folder of original rmg model] ` can allow using 
a starting RMG model. A special path can be added  with the argument `--output` for the
path to output the final files.
"""

import argparse
import logging
import os
import os.path

from rmgpy.rmg.main import initializeLog
from rmgpy.tools.isotopes import run

################################################################################


def parseCommandLineArguments():
    """
    Parse the command-line arguments being passed to RMG-Py. This uses the
    :mod:`argparse` module, which ensures that the command-line arguments are
    sensible, parses them, and returns them.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='RMG input file')
    parser.add_argument('--output', help='Output folder')
    parser.add_argument('--original', help='Location of the isotopeless mechanism')
    args = parser.parse_args()
    
    return args

def main():

    args = parseCommandLineArguments()

    inputFile = args.input
    outputdir = os.path.abspath(args.output) if args.output else os.path.abspath('.')
    original = os.path.abspath(args.original) if args.original else None

    initializeLog(logging.INFO, os.path.join(os.getcwd(), 'RMG.log'))
    run(inputFile, outputdir, original=original)

if __name__ == '__main__':
    main()