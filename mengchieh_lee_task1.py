import os
os.environ["PYSPARK_SUBMIT_ARGS"] = "--packages graphframes:graphframes:0.6.0-spark2.3-s_2.11 pyspark-shell"

from pyspark import SparkContext
import sys
import time
from pyspark.sql import SQLContext
from graphframes import *
from pyspark.sql.types import StructType, StructField, StringType


start = time.time()

sc = SparkContext('local[*]', 'task1')
sqlContext = SQLContext(sc)

filterThreshold = int(sys.argv[1])
filePath = sys.argv[2]

textRDD = sc.textFile(filePath)
header = textRDD.first()
textRDD = textRDD.filter(lambda line: line != header).map(lambda line: line.split(","))


def generate_edge(pair):
    set1 = set(pair[0][1])
    set2 = set(pair[1][1])
    user1 = ""
    user2 = ""
    intersection = set1.intersection(set2)
    if len(intersection) >= filterThreshold:
        user1 = pair[0][0]
        user2 = pair[1][0]

    return [(user1, user2), (user2, user1)]


# (user_id, set(business_id))
userRDD = textRDD.map(lambda arr: (arr[0], arr[1])).groupByKey().mapValues(set)

# construct the list of edges
userPair = userRDD.cartesian(userRDD).filter(lambda t: t[0][0] < t[1][0])
# print("user pair: " + str(userPair.count()))
edgeRDD = userPair.flatMap(generate_edge).filter(lambda edge: edge[0] != edge[1])
# print("edge count: " + str(edgeRDD.count()))

edgeSchema = StructType([StructField("src", StringType()), StructField("dst", StringType())])
edges = sqlContext.createDataFrame(edgeRDD, edgeSchema)
# print(edges.collect())

# add vertices
userOnlyRDD = edgeRDD.flatMap(lambda t: [t[0], t[1]]).distinct().map(lambda t: [t])
# print("vertex count: " + str(userOnlyRDD.count()))
vertexSchema = StructType([StructField("id", StringType())])
vertices = sqlContext.createDataFrame(userOnlyRDD, vertexSchema)
# print(vertices.collect())

g = GraphFrame(vertices, edges)
# print(g)

# do LPA
start2 = time.time()
# (id, label)
result = g.labelPropagation(maxIter=5)
end2 = time.time()
# print('LPA Duration: ' + str(end2 - start2))

communityRDD = result.rdd.map(lambda t: (t[1], t[0]))
sortedCommunity = communityRDD.groupByKey().mapValues(list).map(lambda t: sorted(t[1])).sortBy(lambda t: (len(t), t))
# print(sortedCommunity.top(10))

outputFileName = sys.argv[3]
with open(outputFileName, "w") as fp:
    for c in sortedCommunity.collect():
        fp.write(', '.join(str("\'" + x + "\'") for x in c))
        fp.write("\n")

end = time.time()
print('Duration: ' + str(end - start))
