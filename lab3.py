"""
Dylan O'Hara

dbo8671
"""

import math
from dataclasses import dataclass
import sys
import pickle

@dataclass()
class decTree:

    attr: int
    left: None
    right: None

def entropy(inputs, weights):
    if not inputs: return 0

    pos = 0; neg = 0
    for i, ex in enumerate(inputs):
        if ex[-1] == "en": pos += weights[i]
        else: neg += weights[i]
    q = pos / (pos + neg)
    if q == 1 or q == 0: return 0
    return -(q * math.log2(q) + (1 - q) * math.log2(1 - q))

def create_dt(inputs, vis, depth, weights):

    # # #
    # Create new example sets based on the lowest entropy from the attribute chosen
    # to seperate by (attribute must not be used in a parent node).

    k = depth
    ent = 100
    attr = -1
    l = len(inputs)
    tret = []; fret = []
    tweight = []; fweight = []

    for i in range(8):
        if i in vis:

            t = []; f = []; tw = []; fw = []
            for j, ex in enumerate(inputs):
                if ex[i]: t.append(ex); tw.append(weights[j])
                else: f.append(ex); fw.append(weights[j])

            if (tent := len(t) / l  * entropy(t, tw) + len(f) / l * entropy(f, fw)) < ent:
                ent = tent; attr = i
                tret = t; fret = f
                tweight = tw; fweight = fw

    tvis = vis.copy()
    tvis.remove(attr)

    # Make copy of visited set, temove chosen attribute from new set
    # # #
    # Check base cases

    def basecase():

        depth = k - 1
        if depth == 0:
            pos = 0; neg = 0
            for i, ex in enumerate(tret):
                if ex[-1] == "en": pos += tweight[i]
                else: neg += tweight[i]
            ch = "en" if pos >= neg else "nl"
            left = decTree(ch, None, None)
            pos = 0; neg = 0
            for i, ex in enumerate(fret):
                if ex[-1] == "en": pos += fweight[i]
                else: neg += fweight[i]
            ch = "en" if pos >= neg else "nl"
            right = decTree(ch, None, None)
            return left, right

        ptotal = 0; ntotal = 0
        for i, ex in enumerate(inputs):
            if ex[-1] == "en": ptotal += weights[i]
            else: ntotal += weights[i]

        pos = 0; neg = 0
        for i, ex in enumerate(tret):
            if ex[-1] == "en": pos += tweight[i]
            else: neg += tweight[i]
        if pos == 0: 
            if neg == 0:
                ch = "en" if ptotal >= ntotal else "nl"
                left = decTree(ch, None, None)
            else:
                left = decTree("nl", None, None)
        elif neg == 0: 
            left = decTree("en", None, None)
        else:
            if not tvis:
                ch = "en" if pos >= neg else "nl"
                left = decTree(ch, None, None)
            else:
                left = create_dt(tret, tvis, depth, tweight)

        pos = 0; neg = 0
        for i, ex in enumerate(fret):
            if ex[-1] == "en": pos += fweight[i]
            else: neg += fweight[i]
        if pos == 0: 
            if neg == 0:
                ch = "en" if ptotal >= ntotal else "nl"
                right = decTree(ch, None, None)
            else:
                right = decTree("nl", None, None)
        elif neg == 0: 
            right = decTree("en", None, None)
        else: 
            if not tvis:
                ch = "en" if pos >= neg else "nl"
                right = decTree(ch, None, None)
            else:
                right = create_dt(fret, tvis, depth, fweight)

        return left, right

    left, right = basecase()
    return decTree(attr, left, right)

    # Return decision tree
    # # #

def predict_ens(ens, wei, input):

    en = 0; nl = 0

    for i, dt in enumerate(ens):

        while type(dt.attr) != str:
            if input[dt.attr]: dt = dt.left
            else: dt = dt.right

        if dt.attr == "en": en += wei[i]
        else: nl += wei[i]

    return "en" if en >= nl else "nl"

def adaboost(inputs, treenum, fnum, depth):

    ensemble = []
    ensweight = []
    weights = [1 / len(inputs)] * len(inputs)
    depth = fnum if depth == None else depth

    for i in range(treenum):

        ensemble.append(create_dt(inputs, set([i for i in range(fnum)]), depth, weights))
        error = 0

        for j in range(len(inputs)):
            if predict_ens([ensemble[i]], [1], inputs[j]) != inputs[j][-1]: error += weights[j]
        estat = error / (1 - error)
        for j in range(len(inputs)):
            if predict_ens([ensemble[i]], [1], inputs[j]) == inputs[j][-1]: weights[j] = weights[j] * estat
        
        s = sum(weights)
        for i in range(len(weights)):
            weights[i] = weights[i] / s
        ensweight.append(min(10, math.log(1 / estat)))

    return ensemble, ensweight

def main():

    if sys.argv[1] == "train":

        inputs = []
        with open(sys.argv[2]) as f:

            for line in f:

                # Make lowercase and remove non-alphanumeric characters
                line = line.lower()
                for c in ".,/!'\[{(#%^*@)}]$-_`~:;?<>+=|": line = line.replace(c, " ")
                temp = []
                temp.append(" and " in line)
                temp.append(" the " in line)
                temp.append(" of " in line)
                temp.append(" de " in line)
                temp.append(" het " in line)
                temp.append(" van " in line)
                temp.append((line.count("y") + line.count("c") + line.count("p") + line.count("s")) >= 8)
                temp.append((line.count("ee") + line.count("aa")) < 2)
                temp.append(line[0:2])
                inputs.append(temp)

        stumps = 4 # Trees to use in ensemble
        depth = 6 # Depth of each tree
        fnum = 8 # Number of features

        if sys.argv[4] == "ada":

            ens, wei = adaboost(inputs, stumps, fnum, depth)
            pickle.dump([ens, wei], open(sys.argv[3], "wb"))

        else:

            ens, wei = adaboost(inputs, 1, fnum, depth)
            pickle.dump([ens, wei], open(sys.argv[3], "wb"))

    else:

        inputs = []
        with open(sys.argv[3]) as f:

            for line in f:

                # Make lowercase and remove non-alphanumeric characters
                line = line.lower()
                for c in ".,/!'\[{()}]$-_`~:;?<>+=|": line = line.replace(c, " ")
                temp = []
                temp.append(" and " in line)
                temp.append(" the " in line)
                temp.append(" of " in line)
                temp.append(" de " in line)
                temp.append(" het " in line)
                temp.append(" van " in line)
                temp.append((line.count("y") + line.count("c") + line.count("p") + line.count("s")) >= 8)
                temp.append((line.count("ee") + line.count("aa")) < 2)
                inputs.append(temp)

        d = pickle.load(open(sys.argv[2], "rb"))
        ens = d[0]; wei = d[1]
        for ex in inputs:
            print(predict_ens(ens, wei, ex))

    return

if __name__== "__main__":
    main()