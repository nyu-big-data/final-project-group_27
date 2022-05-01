import numpy as np
import pandas as pd
import code.constants as const
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.window import Window
from pyspark.sql import Row
from pyspark.sql.types import IntegerType


class UnitTest():
    def __init__(self) -> None:
        pass
    # Check to train/val/test splits to make sure approx 60/20/20 split is achieved

    def sanity_check(self, train=None, val=None, test=None):
        """
        Method to print out the shape of train/val/test splits, and a check to make sure that
        val and test splits are disjoint (no distinct userId appears in both)
        input:
        -----
        train: RDD - Training data split created from .create_train_val_test_splits
        val: RDD - Validation data split created from .create_train_val_test_splits
        test: RDD - Testing data split created from .create_train_val_test_splits
        -----
        output:
        -----
        returnFlag: boolean - True means test and val splits are disjoint on userId
        """

        # Get observatio counts for training, val, and test sets
        training_obs, val_obs, test_obs = train.count(), val.count(), test.count()
        total = training_obs + val_obs + test_obs
        # Calculate Percentage Splits
        percent_train = np.round((training_obs/total)*100, 2)
        percent_val = np.round((val_obs/total)*100, 2)
        percent_test = np.round((test_obs/total)*100, 2)

        # Check for Train/Val Leakage
        train_val_check = self.data_leakage_check(train, val)
        if train_val_check == True:
            print(f"Train/Val Leakage Test Passed!")
        # Check for Train/Test Leakage
        train_test_check = self.data_leakage_check(train, test)
        if train_test_check == True:
            print(f"Train/Test Leakage Test Passed!")
        val_test_check = self.data_leakage_check(val, test)
        # Check for Val/Test Leakage
        if val_test_check == True:
            print(f"Val/Test Leakage Test Passed!")

        # Print Stats
        print(
            f"Training Data Len: {training_obs} Val Len: {val_obs}, Test Len: {test_obs}")
        print(
            f"Training {percent_train}%, Val {percent_val}%, Test {percent_test}%")
        print(
            f"Partitions, Train: {train.rdd.getNumPartitions()}, Val: {val.rdd.getNumPartitions()}, Test: {test.rdd.getNumPartitions()}")

        # Need all 3 to evaluate to True
        return train_val_check and train_test_check and val_test_check

    def data_leakage_check(self, train, val):
        """
        Check to make sure that training and validation sets are disjoint on (userId,movieId) pairs
        This function will print out the overlap if there is any - Returns True if the check is passed clean
        """
        # Check if there are any overlapping_ids between Train / Validation
        cond = [train.userId == val.userId, train.movieId ==
                val.movieId]  # Double Join Condition
        # Join Data on Inner - Get Count()
        overllaping_ids = train.join(val, cond, how='inner').count()
        # If there is overlap raise error
        if overllaping_ids != 0:
            overlap = train.join(val, cond, how='inner').select(
                val.userId, val.movieId).take(20)
            overlap = [(x[0], x[1]) for x in overlap]
            print(
                f"First 20 Overlapping movieIds Between Set1 and Set2: {overlap}")
        # Return True if they're disjoint, False if there's overlap
        return overllaping_ids == 0

    def baseline_prediction_check(self, preds):
        m_ids = preds.groupBy("movieId").count().collect()
        m_ids = np.array([int(x[1]) for x in m_ids])
        u_ids = preds.groupBy("userId").count().collect()
        u_ids = np.array([int(x[1]) for x in u_ids])
        if m_ids.sum()/len(u_ids) != len(m_ids):
            raise Exception(
                f"Baseline Predicts are Wrong, movieId count = {m_ids.sum()}, len:{len(m_ids)}")
        if u_ids.sum()/len(m_ids) != len(u_ids):
            raise Exception(
                f"Baseline predictions are wrong, userId count = {u_ids.sum()}, len:{len(u_ids)}")
        print("Passed Baseline Prediction Check")

    def dupe_checker(self, dataFrame):
        """
        Checks for duplicate (userId,movieId) pairs from input DataFrame
        """
        a = dataFrame.select(col("userId"), col("movieId")).collect()
        seen = dict()
        for row in a:
            if (row[0], row[1]) not in seen:
                seen[(row[0], row[1])] = 1
            else:
                print(f"Dupe found at: userId:{row[0]},movieId:{row[1]}")

    def grab_user_rows(self, train, val, pred, userId):
        """
        Grabs all the observations from train, validation, and test sets for a specific userId.
        Input: train,val,pred, userId - All 3 PySpark DataFrames
        output: trains,vals,preds,collected - All 4 are Python lists
        """
        vals = val.select("userId", "movieId", "rating").where(
            col("userId") == userId).collect()
        preds = pred.select("userId", "movieId").where(
            col("userId") == userId).collect()
        trains = train.select("userId", "movieId", "rating").where(
            col("userId") == userId).collect()

        collected = val.select("userId", "movieId").groupBy("userId")\
            .agg(collect_list(col("movieId")).alias("movieId")).where(col("userId") == userId).collect()
        vals = [x[1] for x in vals]
        preds = [x[1] for x in preds]
        trains = [x[1] for x in trains]
        return trains, vals, preds, collected[0][1]
