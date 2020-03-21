from pyspark import SparkContext
import sys
import time

start = time.time()

sc = SparkContext('local[*]', 'task1')

# filePath = "/Users/jamie/PycharmProjects/hw4/ub_sample_test.csv"
filterThreshold = int(sys.argv[1])
filePath = sys.argv[2]

textRDD = sc.textFile(filePath)
header = textRDD.first()
textRDD = textRDD.filter(lambda line: line != header).map(lambda line: line.split(","))

# (user_id, set(business_id))
userRDD = textRDD.map(lambda arr: (arr[0], arr[1])).groupByKey().mapValues(set)


def generate_edge(pair):
    set1 = set(pair[0][1])
    set2 = set(pair[1][1])
    user1 = ""
    user2 = ""
    intersection = set1.intersection(set2)
    if len(intersection) >= filterThreshold:
        user1 = pair[0][0]
        user2 = pair[1][0]

    return user1, user2


def format_edge(normal_edge, current_v):
    u = normal_edge[0]
    if u == current_v:
        u = normal_edge[1]

    return current_v, u


# construct the graph
userPair = userRDD.cartesian(userRDD).filter(lambda t: t[0][0] < t[1][0])
print("user pair: " + str(userPair.count()))
edgeRDD = userPair.map(generate_edge).filter(lambda edge: edge[0] < edge[1])
edgeList = edgeRDD.collect()
print("edge count: " + str(len(edgeList)))
vertexRDD = edgeRDD.flatMap(lambda t: [t[0], t[1]]).distinct()
vertexList = vertexRDD.collect()


def calculate_betweenness(v, tmpEdgeList):
    # print("v is: " + str(v))
    finalList = []

    # level -> list(edge)
    levelMap = dict()
    curLevel = 0
    filterList = list()
    for edge in tmpEdgeList:
        if v in edge:
            filterList.append(edge)
    oldSet = set(filterList)

    orderList = list()
    for edge in filterList:
        if edge[0] != v:
            orderList.append((edge[1], edge[0]))
        else:
            orderList.append(edge)
    levelMap[curLevel] = orderList

    # vertex inDegree
    inDegreeMap = dict()
    inDegreeMap[v] = 1
    for edge in orderList:
        inDegreeMap[edge[1]] = 1

    # do BFS
    indicator = 0
    while indicator != 1:
        nextLevelList = list()
        # (v, u) -> (u, connectedList)
        for curEdge in levelMap[curLevel]:
            u = curEdge[1]
            processEdgeList = list()
            for edge in tmpEdgeList:
                if u in edge and edge not in oldSet:
                    processEdgeList.append(edge)
            # print("filter: " + str(processEdgeList))

            connectedEdgeList = list()
            for edge in processEdgeList:
                if edge[0] != u:
                    connectedEdgeList.append((edge[1], edge[0]))
                else:
                    connectedEdgeList.append(edge)
            # print("ordered: " + str(connectedEdgeList))

            # remove edges with 2 nodes at the same level
            validEdges = list()
            for row in connectedEdgeList:
                exist = False
                for e in levelMap[curLevel]:
                    if row[1] == e[1]:
                        exist = True
                        break
                if not exist:
                    validEdges.append(row)

            # print("valid: " + str(validEdges))
            nextLevelList.extend(validEdges)
            oldSet.update(processEdgeList)

            # (u, xx)
            for k in validEdges:
                xx = k[1]
                if xx in inDegreeMap.keys():
                    inDegreeMap[xx] = inDegreeMap[xx] + inDegreeMap[u]
                else:
                    inDegreeMap[xx] = inDegreeMap[u]
                # if xx not in inDegreeMap.keys():
                #     inDegreeMap[xx] = 0
                # inDegreeMap[xx] = inDegreeMap[xx] + 1

        if len(nextLevelList) == 0:
            indicator = 1
        else:
            curLevel += 1
            levelMap[curLevel] = nextLevelList

    # calculate betweenness
    # node -> betweenness
    nodeValueMap = dict()
    for v in inDegreeMap.keys():
        nodeValueMap[v] = 1

    # from bottom to top
    for i in range(curLevel, -1, -1):
        subList = levelMap[i]
        nodeSet = set()
        for pair in subList:
            nodeSet.add(pair[0])
            nodeSet.add(pair[1])

        for node in nodeSet:
            indegree = inDegreeMap[node]
            for pair in subList:
                # candidate: (u,v), (w,v)
                if pair[1] == node:
                    # print("pair: " + str(pair))
                    betweenness = 1.0 * nodeValueMap[node] * inDegreeMap[pair[0]] / indegree
                    # betweenness = 1.0 * nodeValueMap[node] / indegree
                    # print(str(nodeValueMap[node]) + " / " + str(indegree) + " = " + str(betweenness))
                    nodeValueMap[pair[0]] = nodeValueMap[pair[0]] + betweenness
                    # print("node: " + str(nodeValueMap))

                    lexPair = pair
                    if lexPair[0] > lexPair[1]:
                        lexPair = (pair[1], pair[0])
                    finalList.append((lexPair, betweenness))

    # print("final: " + str(finalList))
    return finalList


betweennessRDD = vertexRDD.flatMap(lambda v: calculate_betweenness(v, edgeList)).reduceByKey(lambda a, b: a+b)\
    .map(lambda t: (t[0], t[1] / 2)).sortBy(lambda t: (-t[1], t[0]))
# for each in betweennessRDD.collect():
#     print(each)

outputFileName = sys.argv[3]
with open(outputFileName, "w") as fp:
    for each in betweennessRDD.collect():
        fp.write(str(each[0]) + ", " + str(each[1]) + "\n")


def detect_island(vertices, edges):
    community_list = list()
    visited = set()

    # do BFS
    for v in vertices:
        if v in visited:
            continue
        community = set()
        community.add(v)
        connectedNode = set()
        for edge in edges:
            if v in edge:
                connectedNode.add(edge[0])
                connectedNode.add(edge[1])
        connectedNode = connectedNode.difference(community)
        community = community.union(connectedNode)

        endFlag = False
        while not endFlag:
            nextLevelNode = set()
            for u in connectedNode:
                for edge in edges:
                    if u in edge:
                        nextLevelNode.add(edge[0])
                        nextLevelNode.add(edge[1])

            nextLevelNode = nextLevelNode.difference(community)

            if len(nextLevelNode) == 0:
                endFlag = True
            else:
                connectedNode = nextLevelNode
                community = community.union(nextLevelNode)

        visited = visited.union(community)
        community_list.append(sorted(list(community)))

    return community_list


# Community detection
m = len(edgeList)

btwList = betweennessRDD.collect()
# print("betweennes count: " + str(len(btwList)))
# print(btwList)

degreeMap = dict()
for edge in edgeList:
    if edge[0] not in degreeMap:
        degreeMap[edge[0]] = 0
    if edge[1] not in degreeMap:
        degreeMap[edge[1]] = 0

    degreeMap[edge[0]] = degreeMap[edge[0]] + 1
    degreeMap[edge[1]] = degreeMap[edge[1]] + 1

afterEdgeList = edgeList.copy()
modularityMap = dict()
preModularity = 0.0
maxModularity = 0.0
communityLen = 0

while len(afterEdgeList) > 0:
    cutEdge = btwList[0][0]
    # print("cut edge: " + str(cutEdge))

    afterEdgeList.remove(cutEdge)
    communities = detect_island(set(vertexList), afterEdgeList)

    # calculate modularity
    modularity = 0.0
    for comm in communities:
        for nodei in comm:
            Ki = degreeMap[nodei]
            for nodej in comm:
                Kj = degreeMap[nodej]
                Aij = 0.0

                if (nodei, nodej) in edgeList or (nodej, nodei) in edgeList:
                    Aij = 1.0
                modularity += Aij - (Ki * Kj * 0.5 / m)

    modularity *= 0.5 / m
    modularityMap[modularity] = communities
    maxModularity = max(maxModularity, modularity)
    # print(str(len(communities)) + "=" + str(modularity))

    reBetweennessRDD = vertexRDD.flatMap(lambda v: calculate_betweenness(v, afterEdgeList)).reduceByKey(lambda a, b: a + b) \
        .map(lambda t: (t[0], t[1] / 2)).sortBy(lambda t: (-t[1], t[0]))
    btwList = reBetweennessRDD.collect()
    # print(btwList)

    # print("modularity: " + str(modularity) + " pre: " + str(preModularity))
    if modularity >= preModularity:
        preModularity = modularity
        print(str(len(communities)) + "=" + str(modularity))
    else:
        break


print("highest: " + str(maxModularity))
print("community count: " + str(len(modularityMap[maxModularity])))
sortedCommunity = sc.parallelize(modularityMap[maxModularity]).map(lambda t: sorted(t)).sortBy(lambda t: (len(t), t))
for x in sortedCommunity.collect():
    print(x)

outputFileName = sys.argv[4]
with open(outputFileName, "w") as fp:
    for c in sortedCommunity.collect():
        fp.write(', '.join(str("\'" + x + "\'") for x in c))
        fp.write("\n")

end = time.time()
print('Duration: ' + str(end - start))
