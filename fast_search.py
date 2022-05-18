from code.Util import Util
import code.constants as const
from code.treeBuilder import TreeBuilder
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.sql.functions import *
from pyspark.sql import Window
import sys
import json

def main(spark, als_filepath, n_trees, search_k):

    test_path = f"{const.HPC_DATA_FILEPATH}als-large-train.csv"
    test = spark.read.csv(test_path)
    #For Large
    als = ALSModel.load(als_filepath)
    rank = als.rank
    #Get User / Item Latent Factors
    item_lfs = als.itemFactors
    user_lfs = als.userFactors

    #New index
    user_lfs = user_lfs.withColumn("index",row_number().over(Window.partitionBy().orderBy("id")))
    item_lfs = item_lfs.withColumn("index",row_number().over(Window.partitionBy().orderBy("id")))

    #Redefine validation data to use index rather than user id
    new_val = user_lfs.join(test, user_lfs.id == test.userId, how='inner').select(user_lfs.id, test.movieId, test.rating)

    tree = TreeBuilder(items=item_lfs, users=user_lfs, rank=rank,n_trees = n_trees, search_k=search_k)
    tree.build_tree()
    tree.get_preds()
    util = Util(preds=tree.preds, val=new_val)
    util.run_metrics()

    tree_vars = vars(tree)
    tree_vars.update(vars(util))

    # Write our results and model parameters
    print(f"Recording the model_params to: {const.RESULTS_SAVE_FILE_PATH}")
    with open(const.RESULTS_SAVE_FILE_PATH, 'a') as output_file:
        output_file.write(json.dumps(tree_vars))
        output_file.write("\n")

if __name__ == "__main__":
    spark = SparkSession.builder.appName('Proj').getOrCreate()
    als_filepath = sys.argv[1]
    n_trees = int(sys.argv[2])
    search_k = int(sys.argv[3])
    main(spark, als_filepath=als_filepath, n_trees=n_trees, search_k=search_k)