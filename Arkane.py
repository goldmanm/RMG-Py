#!/usr/bin/env python3

###############################################################################
#                                                                             #
# RMG - Reaction Mechanism Generator                                          #
#                                                                             #
# Copyright (c) 2002-2019 Prof. William H. Green (whgreen@mit.edu),           #
# Prof. Richard H. West (r.west@neu.edu) and the RMG Team (rmg_dev@mit.edu)   #
#                                                                             #
# Permission is hereby granted, free of charge, to any person obtaining a     #
# copy of this software and associated documentation files (the 'Software'),  #
# to deal in the Software without restriction, including without limitation   #
# the rights to use, copy, modify, merge, publish, distribute, sublicense,    #
# and/or sell copies of the Software, and to permit persons to whom the       #
# Software is furnished to do so, subject to the following conditions:        #
#                                                                             #
# The above copyright notice and this permission notice shall be included in  #
# all copies or substantial portions of the Software.                         #
#                                                                             #
# THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND, EXPRESS OR  #
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,    #
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE #
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER      #
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING     #
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER         #
# DEALINGS IN THE SOFTWARE.                                                   #
#                                                                             #
###############################################################################

"""
This is the main executable script for Arkane, a tool for computing chemical
reaction rates and other properties used in detailed kinetics models using
various methodologies and theories. To run Arkane, use the command ::

    $ python Arkane.py FILE

where ``FILE`` is the path to an Arkane input file describing the job to
execute. Arkane will run the specified job, writing the output to
``output.py`` and a log to both the console and to ``Arkane.log``, with both
files appearing in the same directory as the input file. Some additional
command-line arguments are available; run the command ::

    $ python Arkane.py -h

for more information.
"""

import os
import logging

from arkane.main import Arkane

def run_arkane_commandline():
    arkane = Arkane()

    # Parse and validate the command-line arguments
    arkane.parse_command_line_arguments()

    # Execute the job
    arkane.execute()

    try:
        import psutil

        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        logging.info('Memory used: %.2f MB' % (memory_info.rss / 1024.0 / 1024.0))
    except ImportError:
        logging.info('Optional package dependency "psutil" not found; memory profiling information will not be saved.')

def run_arkane_python(input_file, output_directory=None, verbose=logging.INFO, save_rmg_libraries=True, plot=False):
    """
    method for running arkane from within python (as opposed to from command line)

    Parameters
    ----------
    input_file : str
        string of name of arkane input file
    output_directory : str, optional
        string of directory to save arkane output. The default is None.
    verbose : logging object, optional
        Type of logging used by Arkane. The default is INFO.
    save_rmg_libraries : boolean, optional
        whether or not to save RMG libraries. The default is True.

    Returns
    -------
    None.

    """
    if output_directory and os.path.isdir(output_directory):
        output_directory = os.path.abspath(output_directory)
    else:
        output_directory = os.path.dirname(os.path.abspath(input_file))

    arkane = Arkane(input_file, output_directory, logging.INFO, save_rmg_libraries)
    arkane.plot = plot
    arkane.execute()

if __name__ == '__main__':
    run_arkane_commandline()