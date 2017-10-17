#! /usr/bin/env python
from __future__ import print_function
import commands
import sys

arguments = ' '.join(sys.argv[1:])

cmd = "sbatch -p simulation -c 8 --wrap=\"python {}\"".format(arguments)
status, job_msg = commands.getstatusoutput(cmd)
print(job_msg)
