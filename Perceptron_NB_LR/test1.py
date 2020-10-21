import argparse
import numpy as np
import pandas as pd

def getMaximumScore(A):
    # Write your code here
    score = 0
    op = 1
    
    while(len(A) > 0):
        print(A)
        print(score)
        if op == 1:
            score += sum(A)
            A.remove(max(A[0],A[-1]))
            op = 0
        else:
            score = sum(A)
            A.remove(min(A[0],A[-1]))
            op = 1

           
    
    return score

def main():
    integerarry = [1,3,2]
    print(getMaximumScore(integerarry))

if __name__ == "__main__":
    main()