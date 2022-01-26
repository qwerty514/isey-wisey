import numpy as np
from pathlib import Path

def GetWiki(rowNormalize=True):

    # LOAD in ALLLLL the files
    index = np.loadtxt(Path("data/GReduced_40/GReduced_40/CountryCode.csv"), dtype=object)
    countryPrefs = ["ar", "fr", "de", "en", "ru"]
    matrixTypes = ["GR", "Gpr", "GRR", "Gqr_norm", "Gqr_diag0"]
    tables = {}
    for fn in countryPrefs:
        tables[fn] = {}
        for mt in matrixTypes:
            tables[fn][mt] = np.loadtxt(Path(f"data/GReduced_40/GReduced_40/{fn}_{mt}.txt"))

    # TODO Calculate Weights for other wikis: use decompositions to solve for coeff
    # Waits for EnWiki
    Wrr = 0.009098
    Wqr = 0.029702
    Wpr = 0.96120

    wikiMatrix = (tables["en"]["GRR"] * Wrr) + (tables["en"]["Gqr_diag0"] * Wqr) / (Wqr + Wrr)
    # In case some wiki pages still thought it was useful to refer to themselves
    wikiMatrix *= np.logical_not(np.identity(len(tables["en"]["GR"])))
    wikiMatrix = wikiMatrix.T

    #Row normalize
    if rowNormalize:
        for r in wikiMatrix:
            r /= np.sum(r)
    else:
        avSum = np.average(np.sum(wikiMatrix, axis=1))
        wikiMatrix /= avSum

    sizeIndex = np.sum(wikiMatrix, axis=0)
    #print(sizeIndex)

    return wikiMatrix, index, sizeIndex



def GetComtrade(years, conformationDir="import", cutOff=None, dataCombineMode="union", removeWorld=True, gradient=0, rowNormalize=True):
    ## CHECK PARAMS

    if years == []:
        raise ValueError("No year specified")
    for y in years:
        if y not in range(2010, 2020, 2):
            raise ValueError("No data on this year")
    if conformationDir not in ["import", "export", "gradient"]:
        raise ValueError("Direction doesn't exist")
    if dataCombineMode not in ["union", "intersect"]:
        raise ValueError("Direction doesn't exist")
    if cutOff not in ["sendEq", "sendBySize", "sendWorld", "delete", None]:
        raise ValueError("Invalid cutoffMode option provided")

    ## LOAD FILES

    comtradeDBs = []
    for y in years:
        comtradeDBs.append(np.loadtxt(Path(f"data/COMTRADE/col4.comtrade.{y}.new.dat"), np.uint64))

    ## BUILD INDEX

    index = np.empty(0, np.uint64)

    if dataCombineMode == "union":
        for comtr in comtradeDBs:
            index = np.union1d(index, np.unique(comtr[:, 1]))
            index = np.union1d(index, np.unique(comtr[:, 2]))
    else:
        # No intersect with col2 needed as all in col1 (From) are in col2 (To)
        index = np.unique(comtradeDBs[0][:, 1])
        if len(comtradeDBs) > 1:
            for comtr in comtradeDBs[1:]:
                index = np.intersect1d(index, np.unique(comtr[:, 1]))
                # No intersect with col2 needed as all in col1 (From) are in col2 (To)
                # index = np.intersect1d(index, np.unique(comtr[:, 2]))

    # index.astype(np.uint16, copy=False)
    moneyMatrix = np.zeros((index.size, index.size))
    print(f"Amount of countries before cutting: {index.size}")

    ## BUILD MONEY MATRIX

    for comtr in comtradeDBs:
        for a in comtr:
            # Recipient (2) -> Row
            # Sender (1) -> Column
            # Matrix is always square for simplistic indexing!
            moneyMatrix[np.nonzero(index == int(a[2]))[0][0], np.nonzero(index == int(a[1]))[0][0]] += a[3]

    ## LOOSE ENDS
    looseIndex = (moneyMatrix.sum(axis=0) == 0)
    looseIndex[0] = False   # World should be ignored for now

    # Keep node, pretend it exports
    if cutOff == "sendWorld":
        #print(moneyMatrix.sum(axis=1))

        looseEndReceival = np.multiply(moneyMatrix.sum(axis=1), np.zeros(looseIndex.size, dtype=np.int8), where=np.logical_not(looseIndex))
        #print(looseEndReceival)
        moneyMatrix[0] += looseEndReceival

        index = np.delete(index, looseIndex)
        moneyMatrix = np.delete(moneyMatrix, looseIndex, axis=1)
        moneyMatrix = np.delete(moneyMatrix, looseIndex, axis=0)

    # Delete node, ignore exports to this country
    elif cutOff == "delete":
        index = np.delete(index, looseIndex)
        moneyMatrix = np.delete(moneyMatrix, looseIndex, axis=1)
        moneyMatrix = np.delete(moneyMatrix, looseIndex, axis=0)

    # # Export to every country equally. Only to be used when conformdir == export
    # elif cutOff == "fillEq":
    #     pass  # Not implemented. How much should such a country weigh??
    #     moneyMatrix[:, np.nonzero(moneyMatrix == tcol)[0][0]] += np.ones(tcol.size) / np.sum(tcol)
    #
    # # Same as above, but think of a good distribution?
    # elif cutOff == "fillBySize":
    #     pass  # Not implement
    #
    # # Keep node as it is: no affluence to other nodes (cutOff = None)
    # elif cutOff is None:
    #     pass  # Do Nothing


    if conformationDir != "import":
        moneyMatrix = moneyMatrix.transpose()

    if removeWorld:
        np.delete(moneyMatrix, 0, 0)
        np.delete(moneyMatrix, 0, 1)

    #sizeIndex = moneyMatrix.sum(axis=0)

    # ROW NORMALIZE
    if rowNormalize:
        for row in moneyMatrix:
            try:
                row /= np.sum(row)
            except ZeroDivisionError:
                print("Warning! Node with no neighbours to conform to, check your cutoff + direction settings")
                row = 0
                pass
    else:
        avSum = np.average(np.sum(moneyMatrix, axis=1))
        moneyMatrix /= avSum

    sizeIndex = moneyMatrix.sum(axis=0)


    return moneyMatrix, index, sizeIndex

#
# [moneyMatrix, index, sizeIndex] = GetComtrade([2018, 2016], cutOff="sendWorld")
# print(sizeIndex)
# print(f"Amount of countries after cutting loose ends: {index.size} || {sizeIndex.size}")


# [wikiMatrix, index] = GetWiki()
# print(f"MatrixSZ: {wikiMatrix.size} Index.Size: {index.size}")

