import time
import annoy
from annoy import AnnoyIndex
import pyspark
from pyspark import SparkContext, SparkFiles

def find_neighbors(i):
    from annoy import AnnoyIndex
    ai = AnnoyIndex(250,metric='angular')
    ai.load(SparkFiles.get("tree.ann"))
    #Returns (index,nn list) tuples
    return ((x[2],ai.get_nns_by_vector(vector=x[1], n=100)) for x in i)
class TreeBuilder():
    def __init__(self, items, users, rank, n_trees, search_k):
        self.items = items
        self.users = users
        self.n_trees = n_trees
        self.search_k = search_k
        self.build_time = None
        self.inference_time = None
        self.tree = AnnoyIndex(rank, metric='angular')
        self.preds = None
    
    def run(self):
        self.build_tree()
        self.get_preds()
    
    def build_tree(self):
        start = time.time()
        #Build user Tree
        for row in self.items.rdd.collect():
            #Add index, vecgtor to tree
            self.tree.add_item(row[2],row[1]) #index, vector
        self.tree.build(self.n_trees, n_jobs=-1)
        end = time.time()
        self.build_time = end-start
        #save then load into RAM
        self.tree.save("tree.ann")
        self.tree.load("tree.ann")
        sc.addFile("tree.ann")

    def get_preds(self):
        start = time.time()
        #Start by getting user to user similarities
        columns = ["id","recs"]
        self.preds = self.users.rdd.mapPartitions(find_neighbors).toDF(columns)
        end = time.time()
        self.inference_time = end - start