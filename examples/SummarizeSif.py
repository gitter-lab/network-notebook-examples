from optparse import OptionParser
import KSHVUtil
import os
import sys
import glob
import csv
import itertools
import numpy

__author__ = 'Anthony Gitter'


def main(argList):
    # Parse the arguments, which either come from the command line or a list
    # provided by the Python code calling this function
    parser = CreateParser()
    (options, args) = parser.parse_args(argList)

    if options.indir is None or options.outfile is None:
        raise RuntimeError("Must specify --indir and --outfile")

    if options.pattern is None and options.siflist is None:
        raise RuntimeError("Must specify --pattern or --siflist")

    # Load the common set of prizes
    prizeMap = dict()
    if not options.prizefile is None:
        prizeMap = LoadPrizes(options.prizefile)
        print "%d prizes loaded" % len(prizeMap)
    # The proteins with prizes
    prizes = set(prizeMap.keys())

    # Read and load the information about each forest
    names = []
    forests = []
    forestNodes = []
    forestEdges = []
    connectedComponents = []
    connectedCompSizes = []
    envNeighbors = []
    forestDiameters = []
    maxDegreeNode = []
    maxDegree = []
    
    if not options.pattern is None:
        pattern = os.path.join(options.indir,options.pattern)
        sifFiles = glob.glob(pattern)
    else:
        sifFiles = [os.path.join(options.indir, f) for f in options.siflist.split("|")]
    
    
    for sifFile in sifFiles:
        names.append(os.path.basename(sifFile))
        curForest = KSHVUtil.LoadSifNetwork(sifFile)
        connectedCompNumber = KSHVUtil.numberConnectedComponents(curForest)
        connectedComponents.append(connectedCompNumber)
        sortedSize = KSHVUtil.sortedSizeConnectedComponents(curForest)
        connectedCompSizes.append(sortedSize)
        sortedDiameter = KSHVUtil.sortedDiameterConnectedComponents(curForest, prizes)
        forestDiameters.append(sortedDiameter)
        nodeDegree = KSHVUtil.MaxDegreeNode(curForest)
        maxDegreeNode.append(nodeDegree[0])
        maxDegree.append(nodeDegree[1])
        forests.append(curForest)
        forestNodes.append(set(curForest.nodes()))
        # Sort the nodes in each edge tuple
        forestEdges.append(set(map(SortEdge, curForest.edges())))
        dummyFile = sifFile.replace("optimalForest","dummyForest")
        envNeighbor = []
        with open(dummyFile) as f:
            for line in f.readlines():
                if line.startswith('ENV'):
                    neighbor = line.split('\t')[2] 
                    envNeighbor.append(neighbor)
        envNeighbors.append(envNeighbor)

    print "%d forests loaded" % len(names)
    
    
    #Average size of all forests
    sumForest = 0
    emptyCount = 0
    for forest in forestNodes:
        sumForest = sumForest + len(forest) 
        if(len(forest) == 0):
                emptyCount += 1
    #calculating relative node frequency for all nodes 
    nodeFreq = KSHVUtil.SetFrequency(forestNodes)
    nodeFreqNonEmpty = KSHVUtil.SetFrequencyNonEmpty(forestNodes, emptyCount)
    
    
    #finding the number of low confident, high confident and median of node frequencies for all forests
    lowThreshold = 0.1
    highThreshold = 0.7
    lowFreqNodes = []
    highFreqNodes= []
    medians= []
    
    
    for forest in forests:
        thresholdFreq = KSHVUtil.ApplyThreshold(nodeFreqNonEmpty, forest, lowThreshold, highThreshold)
        lowFreqNodes.append(thresholdFreq[0])
        highFreqNodes.append(thresholdFreq[1])
        medians.append(thresholdFreq[2])
        
    # Store the Steiner nodes, which are the forest nodes that are not prizes
    steinerNodes = []
    for i in range(len(forestNodes)):
        steinerNodes.append(forestNodes[i].difference(prizes))
        
    
        
    # Store the degree of the hub node of interest
    if not options.hubnode is None:
        # The degree is 0 if the hub node is not in the forest
        hubDegrees = [forest.degree(options.hubnode) if options.hubnode in forest else 0 for forest in forests]
        
    # Write the sizes of each tree
    with open(options.outfile + "_size_tmp.txt", "w") as f:
        f.write("Forest name")
        for name in names:
            f.write("\t" + name)
        f.write("\n")
        f.write("Forest size")
        for forest in forestNodes:
            f.write("\t%d" % len(forest))
        f.write("\n")
        # Only write Steiner nodes and prizes if prizes were loaded
        if len(prizes) > 0:
            f.write("Steiner nodes")
            for steiner in steinerNodes:
                f.write("\t%d" % len(steiner))
            f.write("\n")
            f.write("Prizes in forest")
            for i in range(len(forestNodes)):
                f.write("\t%d" % (len(forestNodes[i])-len(steinerNodes[i])))
            f.write("\n")
            f.write("Total prizes")
            for i in range(len(forestNodes)):
                f.write("\t%d" % len(prizes))
            f.write("\n")
            f.write("Connected components")
            for connectedComp in connectedComponents:
                f.write("\t%d" % connectedComp)
            f.write("\n")
            f.write("Size connected components")
            for connectedCompSize in connectedCompSizes:
                f.write("\t%s" % connectedCompSize)
            f.write("\n")
            f.write("Env neighbor")
            for neighbor in envNeighbors:
                f.write("\t%d" % len(neighbor))
            f.write("\n")
            f.write("Largest steiner chain")
            for diameter in forestDiameters:
                f.write("\t%s" % diameter)
            f.write("\n")
            f.write("Low confident genes")
            for lowFreq in lowFreqNodes:
                f.write("\t%s" % lowFreq)
            f.write("\n")
            f.write("High confident genes")
            for highFreq in highFreqNodes:
                f.write("\t%s" % highFreq)
            f.write("\n")
            f.write("Median node frequency")
            for median in medians:
                f.write("\t%s" % median)
            f.write("\n")
            f.write("Max degree node")
            for node in maxDegreeNode:
                f.write("\t%s" % node)
            f.write("\n")
            f.write("Max degree")
            for degree in maxDegree:
                f.write("\t%s" % degree)
            f.write("\n")
        # Only write hub node degrees if a hub was specified and the ratio
        # of hub node degree to forest size
        if not options.hubnode is None:
            f.write("%s degree" % options.hubnode)
            for degree in hubDegrees:
                f.write("\t%d" % degree)
            f.write("\n")
            f.write("%s degree / forest size" % options.hubnode)
            for i in range(len(forestNodes)):
                if (len(forestNodes[i]) == 0):
                    f.write("\t0")
                else:
                    f.write("\t%f" % (float(hubDegrees[i]) / len(forestNodes[i])))
            f.write("\n")
            
    print "%d empty forests" % emptyCount
    print "Average forest size is %d" % (sumForest / float(len(forestNodes)))
    
    # See http://stackoverflow.com/questions/4869189/how-to-pivot-data-in-a-csv-file
    # Transpose the file so that rows become columns
    # The reader iterator iterates over rows and izip is used to create new tuples
    # from the ith element in each row
    with open(options.outfile + "_size_tmp.txt", "rb") as beforeTransFile, open(options.outfile + "_size.txt", "wb") as afterTransFile:
        transposed = itertools.izip(*csv.reader(beforeTransFile, delimiter = '\t'))
        csv.writer(afterTransFile, delimiter = '\t').writerows(transposed)
    # Remove the temporary file
    os.remove(options.outfile + "_size_tmp.txt")

    # Take the union of all nodes in all forests and write the membership of each of them
    #allNodes = set()
    #for forest in forestNodes:
    #    allNodes = allNodes.union(forest)
    #with open(os.path.join(options.workingpath, options.outfile + "_nodes.txt"), "w") as f:
    #    f.write("Node")
    #    for name in names:
    #        f.write("\t" + name)
    #    f.write("\tCount\n")

    #    for node in allNodes:
    #        f.write(node)
    #        count = 0
    #        for i in range(len(prizes)):
    #            if node in steinerNodes[i]:
    #                count += 1
    #                f.write("\tS")
    #            elif node in forestNodes[i]:
    #                count += 1
    #                # A forest node that is not a Steiner node is a prize
    #                f.write("\tP")
    #            else:
    #                f.write("\t")
    #        f.write("\t%d\n" % count)

    # Write a Cytoscape .noa file for the prize frequency
    #with open(options.outfile + "_prizeFreq.noa", "w") as f:
    #    f.write("PrizeFrequency\n")
    #    for (node, freq) in NetworkUtil.SetFrequency(prizes).iteritems():
    #        f.write("%s = %f\n" % (node, freq))
    
    if options.cyto3:
          # Write a Cytoscape attribute table file for the forest node frequency
        with open(options.outfile + "_nodeAnnotation.txt", "w") as f:
            f.write("Protein\tNodeFreq\tPrize\n")
            for (node, freq) in nodeFreq.iteritems():
                f.write("%s\t%f\t%s\n" % (node, freq, prizeMap.setdefault(node, "")))
    
        # Write a Cytoscape attribute table file for the forest edge frequency and a sif file for the union of
        # all forests
        with open(options.outfile + "_edgeAnnotation.txt", "w") as edaFile:
            edaFile.write("Interaction\tEdgeFreq\n")
            with open(options.outfile + "_union.sif", "w") as sifFile:
                for (edge, freq) in KSHVUtil.SetFrequency(forestEdges).iteritems():
                    edaFile.write("%s (pp) %s\t%f\n" % (edge[0], edge[1], freq))
                    sifFile.write("%s pp %s\n" % (edge[0], edge[1]))      
    else:
        # Write a Cytoscape .noa file for the forest node frequency
        with open(options.outfile + "_nodeFreq.noa", "w") as f:
            f.write("NodeFrequency\n")
            for (node, freq) in nodeFreq.iteritems():
                f.write("%s = %f\n" % (node, freq))
    
        # Write a Cytoscape .eda file for the forest edge frequency and a sif file for the union of
        # all forests
        with open(options.outfile + "_edgeFreq.eda", "w") as edaFile:
            with open(options.outfile + "_union.sif", "w") as sifFile:
                edaFile.write("EdgeFrequency\n")
                for (edge, freq) in KSHVUtil.SetFrequency(forestEdges).iteritems():
                    edaFile.write("%s (pp) %s = %f\n" % (edge[0], edge[1], freq))
                    sifFile.write("%s pp %s\n" % (edge[0], edge[1]))


# Sort a pair of nodes alphabetically
def SortEdge(edge):
    return tuple(sorted(edge))


# Load the mapping of all nodes to their prizes, both proteins and mRNAs
# Prizes are not cast as floats but left as strings
def LoadPrizes(prizeFile):
    prizes = dict()
    with open(prizeFile) as f:
        for line in f:
            parts = line.split()
            prizes[parts[0]] = parts[1]
    return prizes

# Setup the option parser
def CreateParser():
    parser = OptionParser()
    parser.add_option("--indir", type="string", dest="indir", help="The path to the directory that contains sif files.", default=None)
    parser.add_option("--pattern", type="string", dest="pattern", help="The filename pattern of the sif files in indir.  Not needed if a siflist is provided instead", default=None)
    parser.add_option("--siflist", type="string", dest="siflist", help="A list of sif files in indir delimited by '|'.  Not used if a pattern is provided.", default=None)
    parser.add_option("--prizefile", type="string", dest="prizefile", help="The path and filename prefix of the prize file (optional).  Assumes the same prize file was used for all prizes.", default=None)
    parser.add_option("--outfile", type="string", dest="outfile", help="The path and filename prefix of the output.  Does not include an extension.", default=None)
    parser.add_option("--hubnode", type="string", dest="hubnode", help="The name of a hub node in the network (optional).  The degree of this node will be reported.", default=None)    
    parser.add_option("--cyto3", action="store_true", dest="cyto3", help="This flag will generate node and edge frequency annotations files in the Cytoscape 3 table format instead of the default Cytoscape 2.8 style.", default=False)
    return parser

if __name__ == "__main__":
    # Use the command line arguments to setup the options (the same as the default OptionParser behavior)
    main(sys.argv[1:])
