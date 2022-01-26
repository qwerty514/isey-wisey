import math

import cupy

from network import *
import numpy as np
import numpy as cp
import networkx as nx
import matplotlib.pyplot as plt
import sys

rng = np.random.default_rng(31613609128)
plt.rcParams['figure.dpi'] = 400
iterations = 50000

netwType = "econ"
cutTies = False

showPhaseChange = False
showNetworks = False
showTcrit = False

showUnDef = True
sampleSize = 5

if cutTies:
    if netwType == "wiki":
        neighbourdegArr = ["All", 2, 5, 10, 20]
    else:
        neighbourdegArr = ["All", 2, 5, 15, 30]
else:
    neighbourdegArr = ["All"]


#########################################################################################

def runMC(s0=0.2, delta=0.5, turmoil=0.1, topo=None, UDArr=None, spins=None):
    sLow = s0 - delta
    sHigh = s0 + delta

    #print("test")
    #runMC.cptopo = cp.asarray(topo)
    if not hasattr(runMC, "extopo"):
        runMC.extopo = topo
        runMC.cptopo = cp.asarray(topo)
    if runMC.extopo is not topo:
        runMC.cptopo = cp.asarray(topo)

    if spins == None:
        # spins = rng.integers(0, 2, len(runMC.cptopo), dtype=cp.bool)
        spins = cp.zeros(len(runMC.cptopo))

    def calcH():
        if not hasattr(calcH, "energySpins"):
            calcH.energySpins = cp.empty(len(spins))
            calcH.downEnergySpins = cp.empty(len(spins))
            calcH.energyMatrix = cp.empty([len(spins), len(spins)])
            calcH.weightedEnergyMatrix = cp.empty_like(calcH.energyMatrix)
        calcH.downEnergySpins = cp.multiply(cp.logical_not(spins), sLow, out=calcH.downEnergySpins)
        calcH.energySpins = cp.multiply(spins, sHigh, out=calcH.energySpins)
        calcH.energySpins = cp.add(calcH.energySpins, calcH.downEnergySpins, out=calcH.energySpins)
        calcH.energyMatrix = cp.outer(calcH.energySpins, calcH.energySpins, out=calcH.energyMatrix)
        calcH.weightedEnergyMatrix = cp.multiply(runMC.cptopo, calcH.energyMatrix, out=calcH.weightedEnergyMatrix)
        return -cp.sum(calcH.weightedEnergyMatrix)

    def calcM():
        if not hasattr(calcH, "energySpins"):
            calcH.energySpins = cp.empty(len(spins))
            calcH.energyMatrix = cp.empty([len(spins), len(spins)])
            calcH.weightedEnergyMatrix = cp.empty_like(calcH.energyMatrix)
            calcH.downEnergySpins = cp.empty(len(spins))
        calcH.energySpins = cp.multiply(spins, sHigh, out=calcH.energySpins)
        calcH.downEnergySpins = cp.multiply(cp.logical_not(spins), sLow, out=calcH.downEnergySpins)
        calcH.energySpins = cp.add(calcH.energySpins, calcH.downEnergySpins, out=calcH.energySpins)
        return cp.sum(calcH.energySpins) / len(calcH.energySpins)

    def boolToColour(coop):
        return "red" if coop else "blue"

    hamilt = calcH()

    for _ in range(iterations):
        curSpin = rng.integers(0, len(runMC.cptopo))
        if UDArr[curSpin]:
            spins[curSpin] = 0
        else:
            spins[curSpin] = not (spins[curSpin])
            propHamilt = calcH()
            if propHamilt <= hamilt:
                hamilt = propHamilt
            elif rng.random() < cp.exp((hamilt - propHamilt) / turmoil):
                hamilt = propHamilt
            else:
                # Press Undo!
                spins[curSpin] = not (spins[curSpin])

    return [calcM(), spins]


def plotNetwork(spins, graphPos):
    plt.figure()
    nx.draw(netGraph, pos=graphPos, node_color=spins, width=edgeWeights, node_size=sizeIndex * 70, arrowsize=5)
    plt.title("Network")
    plt.suptitle(f"{neighbourdeg} Neighbours, T={turmoil}")
    # nx.draw(netGraph, node_color=spins, width=edgeWeights, node_size=10)


#########################################################################################


# Initialize with certain network
if netwType == "wiki":
    adjMat, index, sizeIndex = GetWiki(rowNormalize=True)
else:
    adjMat, index, sizeIndex = GetComtrade([2018], cutOff="sendWorld",
                                             rowNormalize=True, removeWorld=True)

#########################################################################################

# Initialize array for differen T and for results of cooperation
turmoilArr = np.linspace(0.06, 0.4, 250)
posArr = None  # So that node positions in diagram stay consistent throughout cycles

# Try different degrees:
for neighbourdeg in neighbourdegArr:
    # Reset some stuff
    adjMatTrim = np.copy(adjMat)
    resArr = []

    if neighbourdeg != "All":
        for adjRow in range(len(adjMatTrim)):
            besties = np.sort(adjMatTrim[adjRow])
            isBestFriendsArr = (besties[-neighbourdeg] <= adjMatTrim[adjRow])
            adjMatTrim[adjRow] *= isBestFriendsArr
            adjMatTrim[adjRow] /= np.sum(adjMatTrim[adjRow])

    if showNetworks:
        # Plot data about trimmed network
        plt.figure()
        plt.hist(np.sort(np.sum(adjMatTrim, axis=0)), bins=15, label=f"{neighbourdeg} neighbours")
        plt.title(
            f"Weight distribution of countries (nonzero: {np.count_nonzero(np.sum(adjMatTrim, axis=0))} of {len(adjMatTrim[0])})")
        plt.legend()
        plt.xlabel("w(i)")

        # Cut thin ties in visual plotting
        scaledMat = adjMatTrim.copy()
        scaledMat = np.multiply(scaledMat, 0, out=scaledMat, where=(scaledMat < 0.01))
        print(f"MatrixSZ: {adjMatTrim.size} Index.Size: {index.size}")

        netGraph = nx.from_numpy_array(scaledMat, create_using=nx.DiGraph)
        if posArr == None:
            posArr = nx.spring_layout(netGraph)

        edgeWeights = []
        for e in netGraph.edges(data=True):
            edgeWeights.append(e[2]["weight"] * 3)
        print(f"Variance in edgeweights:{np.var(edgeWeights)}")
        if np.var(edgeWeights) > 0.3:
            print("Size difference too big for diagram, sqrt!")
            # edgeWeights = np.sqrt(edgeWeights)
            sizeIndex = np.sqrt(sizeIndex)

    if showPhaseChange:
        for turmoil in turmoilArr:
            res = runMC(0.1, 0.5, turmoil, adjMatTrim)
            resArr.append(res[0])
            if showNetworks:
                if turmoil == turmoilArr[-1]:
                    plotNetwork(res[1], posArr)

        plt.figure("phasechange")
        plt.plot(turmoilArr, resArr, label=f"Degree {neighbourdeg}")

# Plot Results
if showPhaseChange:
    plt.figure("phasechange")
    plt.xlabel("T")
    plt.ylabel("m")
    plt.title("Average spin value in equilibrium")
    plt.legend()
    plt.show()

#########################################################################################

# Run for different S0
if showTcrit:
    critArr = []
    criSArr = []
    sArr = np.logspace(-8, -1, 30) * 2

    turmoilArr = np.linspace(0.01, 0.3, 60)
    for curS0 in sArr:
        for turmoil in turmoilArr:
            res = runMC(curS0, 0.5, turmoil, adjMat)
            if res[0] > 0.46:
                critArr.append(turmoil)
                criSArr.append(curS0)
                break
    print(critArr)
    print(sArr)
    plt.figure("tcrit$")
    plt.plot(criSArr, critArr)
    plt.xscale("log")
    plt.xlabel("$S_0$")
    plt.ylabel("$T_{crit}$")
    plt.show()

#########################################################################################

# Run for different unconditional defectors
if showUnDef:
    plt.figure("pcritall")
    plt.xlim([0, 0.5])
    plt.ylim([0, 0.35])
    plt.xlabel("$p_{ud}$")
    plt.ylabel("$T_{crit}$")
    for curS0 in [0.13, 0.2, 0.3]:
        clitArr = []
        criUnDefArr = []

        nUndefArr = np.linspace(0, round(len(adjMat) / 2), 20, dtype=int)
        # nUndefArr = np.linspace(0, 7, dtype=int)
        for nUnDef in nUndefArr:
            for _ in range(sampleSize):

                # Set up UDs and turmoil sample points
                isUnDefArr = np.zeros(len(adjMat))
                for _ in range(nUnDef % len(adjMat)):
                    isUnDefArr[rng.integers(0, len(adjMat))] = True
                turmoilArr = np.linspace(0.01, 0.3, 60)
                # print(nUnDef)
                # print(isUnDefArr)

                # Find the phase trans
                for turmoil in turmoilArr:
                    res = runMC(s0=curS0, delta=0.5, turmoil=turmoil, topo=adjMat, UDArr=isUnDefArr)
                    if res[0] > 0.46:  # if higher than
                        clitArr.append(turmoil)
                        criUnDefArr.append(nUnDef)
                        break
                print(f"Undef:{nUnDef} / {round(len(adjMat) / 2)} S0: {curS0}")
                sys.stdout.flush()
        criUnDefArr = np.array(criUnDefArr)
        clitArr = np.array(clitArr)
        criUnDefArr = np.divide(criUnDefArr, len(adjMat))
        trend = np.polyfit(criUnDefArr, clitArr, 1)
        trendline = np.poly1d(trend)
        # criUnDefArr[:] = [x / len(adjMat) for x in criUnDefArr]
        print(criUnDefArr)
        plt.figure()
        plt.xlim([0, 0.5])
        plt.ylim([0, 0.3])
        plt.xlabel("$p_{ud}$")
        plt.ylabel("$T_{crit}$")
        plt.scatter(criUnDefArr, clitArr,  s=0.5)
        plt.plot(criUnDefArr, trendline(criUnDefArr))
        plt.figure("pcritall")
        plt.scatter(criUnDefArr, clitArr,  s=0.5, label=f"$S_0$={curS0}")
        plt.plot(criUnDefArr, trendline(criUnDefArr), label=f"$S_0$={curS0}")

    plt.legend()

    plt.show()
