import argparse
import numpy as np
import pandas as pd

def getMaximumScore(arr):
    # Write your code here
    ans = sum(arr)
    dup = float("-inf")
    arr.sort()

    for a in arr:
        if a<= dup:
            ans += 1 + dup - a
            dup += 1
        else: 
            dup = a
    



    
    
    return ans

def main():
    integerarry = [2,6,4,2,1]
    print(getMaximumScore(integerarry))

if __name__ == "__main__":
    main()