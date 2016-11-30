from collections import defaultdict
import math, networkx, os, sys
import numpy as np

__author__ = 'Anthony Gitter'

# Load an interaction network as an undirected graph.
# Each line should have the format "node1 node2 weight" where the presence of
# the weight column is specified by the weight argument.
# There should be no header line in the file.  Can also be used to load
# the "symbol_*" output from the Steiner tree message passing algorithm,
# but all edges will be stored as undirected edges and the dummy node
# will be included.
# TODO Is this function needed anymore?
def LoadNetwork(networkFile, weight=False):
    if weight:
        graph = networkx.read_edgelist(networkFile, delimiter=None, data=(("weight",float),))
    else:
        graph = networkx.read_edgelist(networkFile, delimiter=None)
    #print "Loaded network %s with %d nodes and %d edges" % (networkFile, graph.order(), graph.size())
    return graph


# Load an interaction network as an undirected graph from a graphite
# edge list.  Only the first two columns, the node names, are used.  Edge
# attributes are ignored.  Multiple instances of the same edge are collapsed
# and self edges are ignored.
# TODO This is partly redundant with LoadNetwork
def LoadGraphiteNetwork(networkFile):
    graph = networkx.Graph()
    with open(networkFile) as f:
        # Read all edges in this pathway and build a graph
        # If an edge is listed twice it will only be stored once because this is not a MultiGraph
        for edgeLine in f:
            edgeParts = edgeLine.split()
            # Ignore self edges
            if not edgeParts[0] == edgeParts[1]:
                graph.add_edge(edgeParts[0], edgeParts[1])
    return graph

# Load an interaction network as an undirected graph from a graphite
# edge list.  Only the first and third columns, the node names, are used.  Edge
# attributes are ignored.  Multiple instances of the same edge are collapsed
# and self edges are ignored.  Directions are ignored.
# Edges have the format "node1 edgeType node2"
# TODO need to support directed edges as well
def LoadSifNetwork(networkFile):
    graph = networkx.Graph()
    with open(networkFile) as f:
        # Read all edges in this network and build a graph
        # If an edge is listed twice it will only be stored once because this is not a MultiGraph
        for edgeLine in f:
            edgeParts = edgeLine.split()
            # Ignore self edges
            if not edgeParts[0] == edgeParts[2]:
                graph.add_edge(edgeParts[0], edgeParts[2])
    return graph

# Load an interaciton network as an undirected graph and calculate
# a penalty (negative prize) for each node.  The penalty is
# -mu * degree.  If mu <= 0 then return an empty dictionary.
def DegreePenalties(mu, degPenalty, undirNetworkFile, dirNetworkFile):
    penalties = dict()

    # mu <= 0 means no penalties are desired
    if mu > 0:
        # TODO need to support directed edges as well
        if dirNetworkFile != None and dirNetworkFile != "None":
            raise RuntimeError("Degree penalties for directed networks are not yet supported")
        # Interaction networks are assumed to be weighted
        network = LoadNetwork(undirNetworkFile, True)
        for node in network:
            penalty = -mu * (network.degree(node) ** degPenalty)
            penalties[node] = penalty

    return penalties


# Return the node and edge intersection of two undirected graphs
# Unlike the networkx intersection, the graphs may have different node sets.
def Intersection(graph1, graph2):
    copy = graph1.copy()
    # Iterate through all graph1 edges and remove those that are not in graph2
    for edge in graph1.edges():
        # *edge unpacks the edge tuple
        if not graph2.has_edge(*edge):
            copy.remove_edge(*edge)
    # Iterate through all graph1 nodes and remove those that are not in graph2
    for node in graph1.nodes():
        if not node in graph2:
            copy.remove_node(node)
    return copy


# Take a list of sets and return a dict that maps elements in the sets
# to the number of sets they appear in as ints.  This is not written for
# maximum efficiency but rather readability.
def SetCounts(setList):
    keys = set()
    for curSet in setList:
        keys.update(curSet)

    # Initialize the dictionary that will be used to store the counts of each element
    countDict = dict.fromkeys(keys, 0)
    # Populate the counts
    for curSet in setList:
        for element in curSet:
            countDict[element] += 1

    return countDict


# Take a list of sets and return a dict that maps elements in the sets
# to the fraction of sets they appear in.  This is not written for
# maximum efficiency but rather readability.
def SetFrequency(setList):
    n = float(len(setList)) # Want floating point division
    countDict = SetCounts(setList)

    # Transform the counts into frequencies
    freqDict = {}
    for key in countDict.keys():
        freqDict[key] = countDict[key] / n

    return freqDict

def SetFrequencyNonEmpty(setList,emptyCount):
    n = float(len(setList)) # Want floating point division
    countDict = SetCounts(setList)

    # Transform the counts into frequencies
    freqDict = {}
    for key in countDict.keys():
        freqDict[key] = countDict[key] / (n-emptyCount)

    return freqDict


# Write a collection to a file with one item per line
def WriteCollection(filename, collection):
    with open(filename, "w") as f:
            for item in collection:
                f.write(str(item) + "\n")


# Write a dictionary to a tab-delimited file with one key-value pair
# per line using str() to format the keys and values
def WriteDict(filename, dictionary):
    with open(filename, "w") as f:
            for item in dictionary.iterkeys():
                f.write("%s\t%s\n" % (str(item), str(dictionary[item])))
            
# return the total number of connected components in a forest using networkx
def numberConnectedComponents(forest):
    return networkx.number_connected_components(forest)
                            
# return the sorted sizes of each component in a forest using networkx
def sortedSizeConnectedComponents(forest):
    subGraphs = list(networkx.connected_component_subgraphs(forest))
    size = []    
    for i in range(len(subGraphs)):
        size.append(len(subGraphs[i]))
    return sorted(size)

# return the sorted diameter of each component in a forest using networkx
def sortedDiameterConnectedComponents(forest, prizeNodes):
    graphs = list(networkx.connected_component_subgraphs(forest))
    diametersComp = []
    for i in range(len(graphs)):
        for node in graphs[i].nodes():
            if node in prizeNodes:
                graphs[i].remove_node(node)
    
        if graphs[i]:
            subGraphs = list(networkx.connected_component_subgraphs(graphs[i]))
            diameterList = []
            for i in range(len(subGraphs)):
                diameterList.append(networkx.diameter(subGraphs[i]))
            diametersComp.append(max(diameterList))
    
    return sorted(diametersComp)

# return the low confidence, high confidence and median of node frequency in a forest
def ApplyThreshold(nodeFreqNonEmpty, forest, lowThreshold, highThreshold):
    if forest:
        freq = []
        for node in forest.nodes():
            if nodeFreqNonEmpty.has_key(node):
                freq.append(nodeFreqNonEmpty.get(node))
        frequency = np.array(freq)

        lowFreq =  frequency <= lowThreshold
        highFreq = frequency >= highThreshold
        lowFreq = sum(lowFreq)
        highFreq = sum(highFreq)
        median = np.median(frequency)
        return ([lowFreq, highFreq, median])
    
    else:
        return([0,0,0])


# return the node, degree pair which has maximum degree
def MaxDegreeNode(forest):
    if forest:
        degrees = forest.degree(forest.nodes())
        node = max(degrees, key = degrees.get)
        return ([node, degrees[node]])
    else:
        return ([0,0])


# Load the list of nodes in a three column format PPI network without
# loading the network.  Capitalizes node names.
def LoadNetworkNodes(filename):
    nodes = set()
    with open(filename) as netFile:
        for line in netFile:
            parts = line.upper().strip().split("\t")
            if len(parts) != 3:
                raise RuntimeError("All network lines must have 3 columns\n%s" % line)
            nodes.add(parts[0])
            nodes.add(parts[1])
    return nodes

# Load the nodes with proteomic prizes and clean the names by capitalizing
# them, fixing Excel's SEPT error, and manually cleaning others
def LoadCleanProteomicNodes(filename):
    protPrizes = LoadCleanProteomicPrizes(filename)
    return set(protPrizes.keys())
    
# Load the nodes with proteomic prizes and clean the names by capitalizing
# them, fixing Excel's SEPT error, and manually cleaning others.  Return a
# dict with the K/M score (the score in the KHSV condition relative to the
# mock condition).
# TODO Generalize to load any type of score in the file
def LoadCleanProteomicPrizes(filename):
    protPrizes = dict()
    with open(filename) as protFile:
        # Skip the header
        next(protFile)
        for line in protFile:
            # Have to fix these before trying to parse the spaces in the line
            line = CleanProteomicLine(line)
            parts = line.upper().strip().split(" ")
            if len(parts) != 5:
                raise RuntimeError("All proteomic prize lines must have 5 columns\n%s" % line)
            protein = CleanProteomicId(parts[0])
            # Will generalize this when other types of scores are needed
            prize = parts[1] # K/M score
            protPrizes[protein] = prize
    return protPrizes
    
# Load the TF prize nodes clean the names by capitalizing
# them, fixing Excel's OCT error
def LoadCleanTFNodes(filename):
    nodes = set()
    with open(filename) as tfFile:
        # Skip the header
        next(tfFile)
        for line in tfFile:
            parts = line.upper().strip().split("\t")
            if len(parts) != 3:
                raise RuntimeError("All TF prize lines must have 3 columns\n%s" % line)
            protein = CleanProteomicId(parts[0])
            nodes.add(protein)
    return nodes

def CleanProteomicLine(line):
    """Replace bad ids that contain spaces from lines of the proteomic file"""
    if line.startswith("tr|B4DGC6"):
        print "Fixed node id tr|B4DGC6|B4DGC6_HUMAN"
        return line.replace("tr|B4DGC6|B4DGC6_HUMAN cDNA FLJ58177, highly similar to Transmembrane protein 94 OS=Homo sapiens PE=2 SV=1","tr|B4DGC6|B4DGC6_HUMAN")
    if line.startswith("tr|B7Z7T6"):
        print "Fixed node id tr|B7Z7T6|B7Z7T6_HUMAN"
        return line.replace("tr|B7Z7T6|B7Z7T6_HUMAN cDNA FLJ54712, highly similar to Tight junction protein ZO-2 OS=Homo sapiens PE=2 SV=1","tr|B7Z7T6|B7Z7T6_HUMAN")
    return line
    
def CleanProteomicId(protein):
    """Replace bad protein ids that have been converted to dates"""
    # Special case for broken SEPT genes
    if protein.endswith("-SEP"):
        print "Fixed node id %s" % protein
        return "SEPT" + protein.replace("-SEP","")
    # Special case for broken OCT genes
    if protein.endswith("-OCT"):
        print "Fixed node id %s" % protein
        return "OCT" + protein.replace("-OCT","")
    return protein

def CreatePrizeAnnotations(protFile1, protFile2, phosFile1, phosFile2, outFile):
    """Aggregate prizes from two proteomic mass spec runs and two
    phosphoproteomic runs.  For each protein, determine the max prize across
    all runs, the type of run it occurred in, and whether it is higher in
    KSHV or Mock treatment.  Write annotation file for Cytoscape.
    """
    # Could have written this more elegantly by loading all prizes into a single
    # data structure, but this script won't be reused so it is simple
    (protPrizes1, protConds1) = LoadProteomicPrizesCondition(protFile1)
    kCount = sum([1 for cond in protConds1.values() if cond == "KSHV"])
    mCount = sum([1 for cond in protConds1.values() if cond == "Mock"])
    print "Loaded %d prizes for proteomic run 1.  %d KSHV and %d Mock." % (len(protPrizes1), kCount, mCount)

    (protPrizes2, protConds2) = LoadProteomicPrizesCondition(protFile2)
    kCount = sum([1 for cond in protConds2.values() if cond == "KSHV"])
    mCount = sum([1 for cond in protConds2.values() if cond == "Mock"])
    print "Loaded %d prizes for proteomic run 2.  %d KSHV and %d Mock." % (len(protPrizes2), kCount, mCount)
    
    (phosPrizes1, phosConds1) = LoadProteomicPrizesCondition(phosFile1)
    kCount = sum([1 for cond in phosConds1.values() if cond == "KSHV"])
    mCount = sum([1 for cond in phosConds1.values() if cond == "Mock"])
    print "Loaded %d prizes for phosphoproteomic run 1.  %d KSHV and %d Mock." % (len(phosPrizes1), kCount, mCount)

    (phosPrizes2, phosConds2) = LoadProteomicPrizesCondition(phosFile2)
    kCount = sum([1 for cond in phosConds2.values() if cond == "KSHV"])
    mCount = sum([1 for cond in phosConds2.values() if cond == "Mock"])
    print "Loaded %d prizes for phosphoproteomic run 2.  %d KSHV and %d Mock." % (len(phosPrizes2), kCount, mCount)
    
    allProts = set(protPrizes1.keys()).union(protPrizes2.keys(), phosPrizes1.keys(), phosPrizes2.keys())
    print "%d distinct proteins" % len(allProts)
    
    protCount = 0
    with open(outFile, "w") as f:
        f.write("Protein\tMaxPrize\tMaxPrizeType\tMaxPrizeCondition\tProteomicRun1\tProteomicRun2\tPhoshoproteomicRun1\tPhosphoproteomicRun2\n")
        for prot in sorted(allProts):
            f.write("%s\t" % prot)
            
            # Ensure the max prize is unique
            rankedPrizes = sorted([protPrizes1.setdefault(prot, 0), protPrizes2.setdefault(prot, 0), phosPrizes1.setdefault(prot, 0), phosPrizes2.setdefault(prot, 0)], reverse=True)
            if (rankedPrizes[0] == rankedPrizes[1]):
                raise RuntimeError("%d does not have a unique max prize" % prot)
            maxPrize = rankedPrizes[0]
            f.write("%f\t" % maxPrize)
            
            # Can check each individually now that we know the max prize is unique
            if protPrizes1[prot] == maxPrize:
                f.write("Proteomic\t%s" % protConds1[prot])
                protCount += 1
            if protPrizes2[prot] == maxPrize:
                f.write("Proteomic\t%s" % protConds2[prot])
                protCount += 1
            if phosPrizes1[prot] == maxPrize:
                f.write("Phosphoproteomic\t%s" % phosConds1[prot])
            if phosPrizes2[prot] == maxPrize:
                f.write("Phosphoproteomic\t%s" % phosConds2[prot])
            
            f.write("\t%f" % (protPrizes1[prot] * ConditionSign(protConds1.setdefault(prot,""))))
            f.write("\t%f" % (protPrizes2[prot] * ConditionSign(protConds2.setdefault(prot,""))))
            f.write("\t%f" % (phosPrizes1[prot] * ConditionSign(phosConds1.setdefault(prot,""))))
            f.write("\t%f" % (phosPrizes2[prot] * ConditionSign(phosConds2.setdefault(prot,""))))
            
            f.write("\n")
    print "%d prizes are max in proteomic runs, %d max in phosphoproteomic runs" % (protCount, len(allProts) - protCount)
    
def LoadProteomicPrizesCondition(filename):
    """Return two dicts.  One maps proteins to prizes.  One maps proteins
    to the condition (KSHV or Mock) they were higher in.  The headers
    in these files mislabeled the M and K scores.
    """
    protPrizes = dict()
    protConds = dict()
    with open(filename) as protFile:
        # Skip the header
        next(protFile)
        for line in protFile:
            # Have to fix these before trying to parse the spaces in the line
            line = CleanProteomicLine(line)
            parts = line.upper().strip().split(" ")
            if len(parts) != 5:
                raise RuntimeError("All proteomic prize lines must have 5 columns\n%s" % line)
            protein = CleanProteomicId(parts[0])
            prize = float(parts[1]) # K/M score
            protPrizes[protein] = prize
            log2KM = float(parts[4]) # log2(K/M)
            if log2KM > 0:
                # If the protein is higher in K, verify the M score is 0
                if not parts[3] == "0":
                    raise RuntimeError("KSHV prizes must have M score of 0\n%s" % parts[3])
                protConds[protein] = "KSHV"
            else:
                # If the protein is higher in M, verify the K score is 0
                if not parts[2] == "0":
                    raise RuntimeError("Mock prizes must have K score of 0\n%s" % parts[2])
                protConds[protein] = "Mock"
    return protPrizes, protConds
    
def ConditionSign(condition):
    """Map from conditions to an integer that is used for visualizing KSHV and
    Mock prizes differently in a bar graph
    """
    if condition == "KSHV":
        return 1
    if condition == "Mock":
        return -1
    return 0

def MapMotifNames(motifScoresFile, manualMapFile, hgncMapFile, ambigMapFile, ppiFile, outFile):
    """Merge multiple types of motif name -> gene symbol mappings and track which
    is used to map a name.  Also create a single motif score and track whether
    the TF is more active in KSHV or Mock treatment conditions.
    """
    # Load the motif scores
    motifCondMap = dict()
    motifScoreMap = dict()
    with open(motifScoresFile) as motifF:
        # Skip the header
        next(motifF)
        for line in motifF:
            parts = line.upper().strip().split("\t")
            assert len(parts) == 3, "Motif score line must have three columns"
            motif = CleanProteomicId(parts[0])
            kScore = float(parts[1])
            mScore = float(parts[2])
            assert kScore != mScore, "Assume motif has a higher score in one condition"
            # Initially set the score to be the sum of the scores in the two conditions
            motifScoreMap[motif] = kScore + mScore
            if kScore > mScore:
                motifCondMap[motif] = "KSHV"
            else:
                motifCondMap[motif] = "Mock"
    
    # Subtract the min score from all scores
    minMotifScore = min(motifScoreMap.values())
    motifScoreMap.update({motif : score-minMotifScore for (motif, score) in motifScoreMap.iteritems()})
    # Divide by the new max score
    maxMotifScore = max(motifScoreMap.values())
    motifScoreMap.update({motif : score/maxMotifScore for (motif, score) in motifScoreMap.iteritems()})
    
    # Track which motif names still need to be mapped.  The mapping files
    # are prioritized such that if there is a good match, then we will
    # ignore other possible lower-quality matches.
    unmappedMotifs = set(motifCondMap.keys())
    assert len(unmappedMotifs) == len(motifCondMap), "Assume unique motif names in score file"
    motifNameMap = dict()
    motifEvMap = dict() # Track the evidence used to map the motif name
    convertedCounter = 0
    
    # First convert motif names using the manual mapping file
    print "Manual mapping file: %s" % os.path.basename(manualMapFile)
    with open(manualMapFile) as manualF:
        # Skip the header
        next(manualF)
        for line in manualF:
            parts = line.upper().strip().split("\t")
            assert len(parts) >= 2, "Every line must contain at least one gene symbol"
            motif = parts[0]
            genes = set(parts[1:]) # May have multiple matches for gene families or complexes
            assert motif in unmappedMotifs, "Manually mapped motifs must be in the motif score file"
            motifNameMap[motif] = genes
            motifEvMap[motif] = "Manual"
            unmappedMotifs.remove(motif)
            convertedCounter += 1
    print "Converted %d motifs with manual mapping file" % convertedCounter
    convertedCounter = 0
    
    # Next identify motifs that are already gene symbols in the network
    print "Network file: %s" % os.path.basename(ppiFile)
    networkGenes = set()
    with open(ppiFile) as ppiF:
        # No header
        for line in ppiF:
            parts = line.upper().strip().split("\t")
            assert len(parts) >= 3, "Every line must contain two genes and a weight"
            networkGenes.add(parts[0])
            networkGenes.add(parts[1])
    print "Loaded %d genes from the PPI network" % len(networkGenes)
    # Iterate over a copy of the unmapped motifs so they be removed from
    # the map during the iteration
    for motif in list(unmappedMotifs):
        if motif in networkGenes:
            motifNameMap[motif] = (motif,)
            motifEvMap[motif] = "Network"
            unmappedMotifs.remove(motif)
            convertedCounter += 1
    print "Converted %d motifs with network file" % convertedCounter
    convertedCounter = 0
    
    # Next use the ambiguous HGNC matches that were manually resolved
    print "Ambiguous HGNC mapping file: %s" % os.path.basename(ambigMapFile)
    with open(ambigMapFile) as ambigF:
        # Skip the header
        next(ambigF)
        for line in ambigF:
            parts = line.upper().strip().split("\t")
            assert len(parts) == 7, "Every line must contain 7 columns"
            motif = parts[0]
            gene = parts[2]
            if motif in unmappedMotifs:
                motifNameMap[motif] = (gene,)
                motifEvMap[motif] = "AmbiguousHGNC"
                unmappedMotifs.remove(motif)
                convertedCounter += 1
    print "Converted %d motifs with ambiguous HGNC mapping file" % convertedCounter
    convertedCounter = 0
    
    # Next use the automatic, unambiguous HGNC mapping
    print "Automatic HGNC mapping file: %s" % os.path.basename(hgncMapFile)
    with open(hgncMapFile) as hgncF:
        # Skip the header
        next(hgncF)
        for line in hgncF:
            parts = line.upper().strip().split("\t")
            if not parts[1] == "UNMATCHED":
                assert len(parts) == 6, "Every line must contain 7 columns"
                motif = parts[0]
                gene = parts[2]
                if motif in unmappedMotifs:
                    motifNameMap[motif] = (gene,)
                    motifEvMap[motif] = "AutomaticHGNC"
                    unmappedMotifs.remove(motif)
                    convertedCounter += 1
    print "Converted %d motifs with automatic HGNC mapping file" % convertedCounter
    convertedCounter = 0
    
    # Finally store motifs that could not be mapped
    print "%d motifs could not be mapped" % len(unmappedMotifs)
    for motif in unmappedMotifs:
        motifNameMap[motif] = (motif,)
        motifEvMap[motif] = "Unmapped"
        
    assert motifScoreMap.viewkeys() == motifNameMap.viewkeys()
    
    # Write all motif-gene symbols mappings to the output file
    with open(outFile, "w") as outF:
        outF.write("Motif\tGeneSymbol\tPrize\tCondition\tMappingEvidence\n")
        for motif in sorted(motifScoreMap.iterkeys()):
            for gene in sorted(motifNameMap[motif]):
                outF.write("%s\t%s\t%f\t%s\t%s\n" % (motif, gene, motifScoreMap[motif], motifCondMap[motif], motifEvMap[motif]))
        
    return motifScoreMap

def MergeProtTFPrizes(protPrizeFile, tfPrizeFile, mergedFile):
    """Merge proteomic prizes that have already been combined by taking the max
    over multiple (phospho)proteomic runs and the TF prizes.  TF prizes still
    may contain multiple prizes per gene so the function takes the max over
    all such prizes.  Both types of prizes have already been normalized to be
    in [0, 1] and we assume that no further rescaling is needed.
    """
    # Load the proteomic prizes
    genePrizes = defaultdict(float)
    with open(protPrizeFile) as protPrizeF:
        for line in protPrizeF:
            # No header
            parts = line.upper().strip().split("\t")
            assert len(parts) == 2, "Every line must contain 2 columns"
            genePrizes[parts[0]] = float(parts[1])
    print "Loaded %d proteomic prizes" % len(genePrizes)
    
    # Load the TF prizes, taking the max for each gene.  Include unmapped
    # motifs because the PPI network could change in the future.
    tfPrizes = defaultdict(float)
    with open(tfPrizeFile) as tfPrizeF:
        # Skip the header
        next(tfPrizeF)
        for line in tfPrizeF:
            parts = line.upper().strip().split("\t")
            assert len(parts) == 5, "Every line must contain 5 columns"
            # Default prize value is 0.0
            tf = parts[1] # Gene symbol, not the motif
            tfPrizes[tf] = max(tfPrizes[tf], float(parts[2]))
    print "Loaded %d prizes for unique TFs" % len(tfPrizes)
    
    # Combine and write the prizes
    for tf, tfPrize in tfPrizes.iteritems():
        genePrizes[tf] = max(genePrizes[tf], tfPrize)
    print "%d prizes after merging protoemic and TF prizes" % len(genePrizes)
        
    with open(mergedFile, "w") as outF:
        for gene in sorted(genePrizes.iterkeys()):
            outF.write("%s\t%f\n" % (gene, genePrizes[gene])) 
            
def CreateProtTFPrizeAnnotations(protFile1, protFile2, phosFile1, phosFile2, tfFile, outFile):
    """Aggregate prizes from two proteomic mass spec runs, two
    phosphoproteomic runs, and the TF motif scores.  For each protein,
    determine the max prize across all runs, the type of run it occurred in,
    and whether it is higher in KSHV or Mock treatment.  The TF motif scores
    can have multiple prizes for a single protein, which is accounted for
    by the function that loads the prizes.
    Writes an annotation file for Cytoscape.
    """
    # Could have written this more elegantly by loading all prizes into a single
    # data structure, but this script won't be reused so it is simple
    (protPrizes1, protConds1) = LoadProteomicPrizesCondition(protFile1)
    kCount = sum([1 for cond in protConds1.values() if cond == "KSHV"])
    mCount = sum([1 for cond in protConds1.values() if cond == "Mock"])
    print "Loaded %d prizes for proteomic run 1.  %d KSHV and %d Mock." % (len(protPrizes1), kCount, mCount)

    (protPrizes2, protConds2) = LoadProteomicPrizesCondition(protFile2)
    kCount = sum([1 for cond in protConds2.values() if cond == "KSHV"])
    mCount = sum([1 for cond in protConds2.values() if cond == "Mock"])
    print "Loaded %d prizes for proteomic run 2.  %d KSHV and %d Mock." % (len(protPrizes2), kCount, mCount)
    
    (phosPrizes1, phosConds1) = LoadProteomicPrizesCondition(phosFile1)
    kCount = sum([1 for cond in phosConds1.values() if cond == "KSHV"])
    mCount = sum([1 for cond in phosConds1.values() if cond == "Mock"])
    print "Loaded %d prizes for phosphoproteomic run 1.  %d KSHV and %d Mock." % (len(phosPrizes1), kCount, mCount)

    (phosPrizes2, phosConds2) = LoadProteomicPrizesCondition(phosFile2)
    kCount = sum([1 for cond in phosConds2.values() if cond == "KSHV"])
    mCount = sum([1 for cond in phosConds2.values() if cond == "Mock"])
    print "Loaded %d prizes for phosphoproteomic run 2.  %d KSHV and %d Mock." % (len(phosPrizes2), kCount, mCount)
    
    (tfPrizes, tfConds) = LoadTFPrizesCondition(tfFile)
    kCount = sum([1 for cond in tfConds.values() if cond == "KSHV"])
    mCount = sum([1 for cond in tfConds.values() if cond == "Mock"])
    print "Loaded %d prizes for TFs.  %d KSHV and %d Mock." % (len(tfPrizes), kCount, mCount)
    
    allProts = set(protPrizes1.keys()).union(protPrizes2.keys(), phosPrizes1.keys(), phosPrizes2.keys(), tfPrizes.keys())
    print "%d distinct proteins" % len(allProts)
    
    protCount = 0
    phosCount = 0
    tfCount = 0
    with open(outFile, "w") as f:
        f.write("Protein\tMaxPrize\tMaxPrizeType\tMaxPrizeCondition\tProteomicRun1\tProteomicRun2\tPhoshoproteomicRun1\tPhosphoproteomicRun2\tTF\n")
        for prot in sorted(allProts):
            f.write("%s\t" % prot)
            
            # Ensure the max prize is unique
            rankedPrizes = sorted([protPrizes1.setdefault(prot, 0), protPrizes2.setdefault(prot, 0), phosPrizes1.setdefault(prot, 0), phosPrizes2.setdefault(prot, 0), tfPrizes.setdefault(prot, 0)], reverse=True)
            if (rankedPrizes[0] == rankedPrizes[1]):
                raise RuntimeError("%s does not have a unique max prize" % prot)
            maxPrize = rankedPrizes[0]
            f.write("%f\t" % maxPrize)
            
            # Can check each individually now that we know the max prize is unique
            if protPrizes1[prot] == maxPrize:
                f.write("Proteomic\t%s" % protConds1[prot])
                protCount += 1
            if protPrizes2[prot] == maxPrize:
                f.write("Proteomic\t%s" % protConds2[prot])
                protCount += 1
            if phosPrizes1[prot] == maxPrize:
                f.write("Phosphoproteomic\t%s" % phosConds1[prot])
                phosCount += 1
            if phosPrizes2[prot] == maxPrize:
                f.write("Phosphoproteomic\t%s" % phosConds2[prot])
                phosCount += 1
            if tfPrizes[prot] == maxPrize:
                f.write("TF\t%s" % tfConds[prot])
                tfCount += 1
            
            f.write("\t%f" % (protPrizes1[prot] * ConditionSign(protConds1.setdefault(prot,""))))
            f.write("\t%f" % (protPrizes2[prot] * ConditionSign(protConds2.setdefault(prot,""))))
            f.write("\t%f" % (phosPrizes1[prot] * ConditionSign(phosConds1.setdefault(prot,""))))
            f.write("\t%f" % (phosPrizes2[prot] * ConditionSign(phosConds2.setdefault(prot,""))))
            f.write("\t%f" % (tfPrizes[prot] * ConditionSign(tfConds.setdefault(prot,""))))
            
            f.write("\n")
    print "%d prizes are max in proteomic runs, %d max in phosphoproteomic runs, %d max in TF" % (protCount, phosCount, tfCount)
    
def LoadTFPrizesCondition(filename):
    """Return two dicts.  One maps proteins to prizes.  One maps proteins
    to the condition (KSHV or Mock) they were higher in.  Proteins may be
    listed more than one time in the file.  Because TF prizes were scaled to be
    between 0 and 1, at least one prize will be 0.0.  Set it to be some
    small epsilon instead for visualization purposes.
    """
    protPrizes = defaultdict(float)
    protConds = dict()
    with open(filename) as tfFile:
        # Skip the header
        next(tfFile)
        for line in tfFile:
            # Don't need to clean ids because that was already done when
            # this file was created
            parts = line.strip().split("\t")
            assert len(parts) == 5, "Every line must contain 5 columns"
            # Default prize value is 0.0
            tf = parts[1] # Gene symbol, not the motif
            # Use the min float as epsilon
            prize = max(float(parts[2]), sys.float_info.min)
            if prize > protPrizes[tf]:
                protPrizes[tf] = prize
                # Track the condition that is associated with this max prize
                protConds[tf] = parts[3]
                
    return protPrizes, protConds

def PrepENSGTFPrizes(tfScoreFilename, ensgFilename, outFilename):
    """Write a new file containing prizes for PCSF for each TF entry.
    A TF may be listed multiple times.  Convert to gene sybmols from
    Ensemble gene ids, which are more reliable than the TF names in
    the input file.  The prize is -log10(q-value).
    """
    # Load the Ensemble gene id to gene symbol mapping
    ensgMap = dict()
    with open(ensgFilename) as ensgFile:
        # Skip the header
        next(ensgFile)
        for line in ensgFile:
            parts = line.strip().upper().split("\t")
            assert len(parts) == 3, "Every line must contain 3 ids"
            # Don't need the HGNC id at this point
            ensgMap[parts[0]] = parts[2]
    
    print "Loaded %d Ensembl Gene ID mappings" % len(ensgMap)
    
    # Scan through the TF score file, convert ids, create prizes
    tfs = 0
    with open(outFilename, "w") as f:
        f.write("TF\tPrize\n")
        
        with open(tfScoreFilename) as tfScoreFile:
            header = next(tfScoreFile).split("\t")
            assert header[5] == "q.value", "q.value column missing"
            assert header[6] == "EnsembleID", "EnsembleID column missing"
            
            for line in tfScoreFile:
                tfs += 1
                parts = line.strip().split("\t")
                qval = float(parts[5])
                prize = -math.log10(qval)
                
                ensgId = parts[6]
                assert ensgId in ensgMap, "Do not recognize %s" % ensgId
                tf = ensgMap[ensgId]
                
                f.write("%s\t%f\n" % (tf, prize))
    print "Wrote scores for %d TFs" % tfs

def MergeProtENSGTFPrizes(protPrizeFile, tfPrizeFile, mergedFile):
    """Merge proteomic prizes, which have already been combined by taking the max
    over multiple (phospho)proteomic runs, and the TF prizes.  TF prizes still
    may contain multiple prizes per gene so the function takes the max over
    all such prizes.  We assume that no rescaling is needed for either type
    of prize, though they are typically not on the same scale.  Proteomic
    prizes have been scaled to [0,1] and TF prizes are unbounded -log10(qvalue).
    Returns the merged prizes that are written to file.
    """
    # Load the proteomic prizes
    genePrizes = defaultdict(float)
    with open(protPrizeFile) as protPrizeF:
        for line in protPrizeF:
            # No header
            parts = line.upper().strip().split("\t")
            assert len(parts) == 2, "Every line must contain 2 columns"
            genePrizes[parts[0]] = float(parts[1])
    numProt = len(genePrizes)
    print "Loaded %d proteomic prizes" % numProt
    
    # Load the TF prizes, taking the max for each gene
    # Could do this without creating a new dict but want to verify
    # how many TF prizes and unique TFs there are
    tfPrizes = defaultdict(float)
    tfLines = 0
    with open(tfPrizeFile) as tfPrizeF:
        # Skip the header
        next(tfPrizeF)
        for line in tfPrizeF:
            tfLines += 1
            parts = line.upper().strip().split("\t")
            assert len(parts) == 2, "Every line must contain 2 columns"
            # Default prize value is 0.0
            tf = parts[0] # Gene symbol
            tfPrizes[tf] = max(tfPrizes[tf], float(parts[1]))
    numTf = len(tfPrizes)
    print "Loaded %d prizes for %d unique TFs" % (tfLines, numTf)  
    
    # Combine and write the prizes
    for tf, tfPrize in tfPrizes.iteritems():
        genePrizes[tf] = max(genePrizes[tf], tfPrize)
    print "%d prizes after merging protoemic and TF prizes" % len(genePrizes)
    print "%d genes with both proteomic and TF prizes" % (numProt + numTf - len(genePrizes)) 
        
    with open(mergedFile, "w") as f:
        for gene in sorted(genePrizes.iterkeys()):
            f.write("%s\t%f\n" % (gene, genePrizes[gene])) 
    
    return genePrizes
    
def CreateProtENSGTFPrizeAnnotations(protFile1, protFile2, phosFile1, phosFile2, tfFile, ensgFile, outFile):
    """Aggregate prizes from two proteomic mass spec runs, two
    phosphoproteomic runs, and the TF motif scores (Ensembl gene id version).
    For each protein, determine the max prize across all runs, the type
    of run it occurred in, and whether it is higher in KSHV or Mock treatment.
    The input TF motif scores can have multiple prizes for a single protein, which is
    accounted for by the function that loads the prizes.  The Ensembl gene id
    mapping is needed to convert the TF identifiers to standard gene symbols.
    Writes an annotation file for Cytoscape.
    """
    # Could have written this more elegantly by loading all prizes into a single
    # data structure, but this script won't be reused so it is simple
    (protPrizes1, protConds1) = LoadProteomicPrizesCondition(protFile1)
    kCount = sum([1 for cond in protConds1.values() if cond == "KSHV"])
    mCount = sum([1 for cond in protConds1.values() if cond == "Mock"])
    print "Loaded %d prizes for proteomic run 1.  %d KSHV and %d Mock." % (len(protPrizes1), kCount, mCount)

    (protPrizes2, protConds2) = LoadProteomicPrizesCondition(protFile2)
    kCount = sum([1 for cond in protConds2.values() if cond == "KSHV"])
    mCount = sum([1 for cond in protConds2.values() if cond == "Mock"])
    print "Loaded %d prizes for proteomic run 2.  %d KSHV and %d Mock." % (len(protPrizes2), kCount, mCount)
    
    (phosPrizes1, phosConds1) = LoadProteomicPrizesCondition(phosFile1)
    kCount = sum([1 for cond in phosConds1.values() if cond == "KSHV"])
    mCount = sum([1 for cond in phosConds1.values() if cond == "Mock"])
    print "Loaded %d prizes for phosphoproteomic run 1.  %d KSHV and %d Mock." % (len(phosPrizes1), kCount, mCount)

    (phosPrizes2, phosConds2) = LoadProteomicPrizesCondition(phosFile2)
    kCount = sum([1 for cond in phosConds2.values() if cond == "KSHV"])
    mCount = sum([1 for cond in phosConds2.values() if cond == "Mock"])
    print "Loaded %d prizes for phosphoproteomic run 2.  %d KSHV and %d Mock." % (len(phosPrizes2), kCount, mCount)
    
    (tfPrizes, tfConds) = LoadENSGTFPrizesCondition(tfFile, ensgFile)
    kCount = sum([1 for cond in tfConds.values() if cond == "KSHV"])
    mCount = sum([1 for cond in tfConds.values() if cond == "Mock"])
    print "Loaded %d prizes for TFs.  %d KSHV and %d Mock." % (len(tfPrizes), kCount, mCount)
    
    allProts = set(protPrizes1.keys()).union(protPrizes2.keys(), phosPrizes1.keys(), phosPrizes2.keys(), tfPrizes.keys())
    print "%d distinct proteins" % len(allProts)
    
    protCount = 0
    phosCount = 0
    tfCount = 0
    with open(outFile, "w") as f:
        f.write("Protein\tMaxPrize\tMaxPrizeType\tMaxPrizeCondition\tProteomicRun1\tProteomicRun2\tPhoshoproteomicRun1\tPhosphoproteomicRun2\tTF\n")
        for prot in sorted(allProts):
            f.write("%s\t" % prot)
            
            # Ensure the max prize is unique
            rankedPrizes = sorted([protPrizes1.setdefault(prot, 0), protPrizes2.setdefault(prot, 0), phosPrizes1.setdefault(prot, 0), phosPrizes2.setdefault(prot, 0), tfPrizes.setdefault(prot, 0)], reverse=True)
            if (rankedPrizes[0] == rankedPrizes[1]):
                raise RuntimeError("%s does not have a unique max prize" % prot)
            maxPrize = rankedPrizes[0]
            f.write("%f\t" % maxPrize)
            
            # Can check each individually now that we know the max prize is unique
            if protPrizes1[prot] == maxPrize:
                f.write("Proteomic\t%s" % protConds1[prot])
                protCount += 1
            if protPrizes2[prot] == maxPrize:
                f.write("Proteomic\t%s" % protConds2[prot])
                protCount += 1
            if phosPrizes1[prot] == maxPrize:
                f.write("Phosphoproteomic\t%s" % phosConds1[prot])
                phosCount += 1
            if phosPrizes2[prot] == maxPrize:
                f.write("Phosphoproteomic\t%s" % phosConds2[prot])
                phosCount += 1
            if tfPrizes[prot] == maxPrize:
                f.write("TF\t%s" % tfConds[prot])
                tfCount += 1
            
            f.write("\t%f" % (protPrizes1[prot] * ConditionSign(protConds1.setdefault(prot,""))))
            f.write("\t%f" % (protPrizes2[prot] * ConditionSign(protConds2.setdefault(prot,""))))
            f.write("\t%f" % (phosPrizes1[prot] * ConditionSign(phosConds1.setdefault(prot,""))))
            f.write("\t%f" % (phosPrizes2[prot] * ConditionSign(phosConds2.setdefault(prot,""))))
            f.write("\t%f" % (tfPrizes[prot] * ConditionSign(tfConds.setdefault(prot,""))))
            
            f.write("\n")
    print "%d prizes are max in proteomic runs, %d max in phosphoproteomic runs, %d max in TF" % (protCount, phosCount, tfCount)

def LoadENSGTFPrizesCondition(tfFilename, ensgFilename):
    """Return two dicts.  One maps proteins to prizes.  One maps proteins
    to the condition (KSHV or Mock) they were higher in, which is determined
    using the sign of the z-score.  Proteins may be
    listed more than one time in the file.  Ensemble gene ids are converted
    to gene symbols for each TF.  The prizes are -log10(q-value).
    """
    # Load the Ensemble gene id to gene symbol mapping
    ensgMap = dict()
    with open(ensgFilename) as ensgFile:
        # Skip the header
        next(ensgFile)
        for line in ensgFile:
            parts = line.strip().upper().split("\t")
            assert len(parts) == 3, "Every line must contain 3 ids"
            # Don't need the HGNC id at this point
            ensgMap[parts[0]] = parts[2]
    
    print "Loaded %d Ensembl Gene ID mappings" % len(ensgMap)
    
    # Scan through the TF score file, convert ids, create prizes, store condition
    # Default prize value is 0.0
    protPrizes = defaultdict(float)
    protConds = dict()
    with open(tfFilename) as tfScoreFile:
        header = next(tfScoreFile).split("\t")
        assert header[3] == "z.score", "z.score column missing"
        assert header[5] == "q.value", "q.value column missing"
        assert header[6] == "EnsembleID", "EnsembleID column missing"
        
        for line in tfScoreFile:
            parts = line.strip().split("\t")
            assert len(parts) == 17, "Every line must contain 17 columns"

            zscore = float(parts[3])
            if zscore > 0:
                cond = "KSHV"
            else:
                cond = "Mock"
            
            qval = float(parts[5])
            prize = -math.log10(qval)

            ensgId = parts[6]
            assert ensgId in ensgMap, "Do not recognize %s" % ensgId
            tf = ensgMap[ensgId]

            # Track the condition that is associated with this max prize
            if prize > protPrizes[tf]:
                protPrizes[tf] = prize
                protConds[tf] = cond
                
    return protPrizes, protConds
    
def ParseWebGestalt(filename):
    """Return a dict mapping KEGG pathway names to the set of genes
    in that enriched pathway and a second dict mapping genes to lists of
    KEGG pathway names.  Assumes all pathways in the file are
    significant.
    """
    pathwayMap = dict()
    geneMap = defaultdict(list)
    with open(filename) as f:
        # Read the entire file into memory
        contents = f.read()
        # Remove trailing newlines, then break into blocks
        fileBlocks = contents.strip().split('\n\n\n')
        
        # First block is header information
        fileBlocks = fileBlocks[1:]
        for block in fileBlocks:
            block = block.replace('\n\n', '\n')
            lines = block.split('\n')
            # First line is the KEGG pathway
            pathway = lines[0].replace('\t', ';')
            # Next line is enrichment statistics and can be ignored
            genes = set()
            for line in lines[2:]:
                # Each line is now a gene in the enriched pathway
                # Keep the uploaded id in the first column
                parts = line.split('\t')
                gene = parts[0]
                genes.add(gene)
                geneMap[gene].append(pathway)
            print '%s had %d genes' % (pathway, len(genes))
            pathwayMap[pathway] = genes
    
    print 'Parsed %d pathways' % len(pathwayMap)
    print 'Found %d genes' % len(geneMap)
    return pathwayMap, geneMap
            