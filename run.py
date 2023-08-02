"""
@author: natnij
"""
import sys
from simpleor.solutions.assignment import runAssignment
from simpleor.solutions.jobshopScheduling import runJobshopScheduling

if __name__ == "__main__":
    arg = sys.argv[1]
    if arg == "assignment":
        runAssignment()
    elif arg == "jobshop":
        runJobshopScheduling()
