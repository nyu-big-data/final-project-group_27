"""
A python file to import constants that point to data sources
"""
HPC_DATA_FILEPATH = "/scratch/work/courses/DSGA1004-2021/movielens/"
CSV_NAME_LIST = ['movies.csv','ratings.csv','tags.csv','links.csv']
SMALL = 'hm-latest-small'
LARGE = 'hm-latest'
data_set_dict = {'small': HPC_DATA_FILEPATH+SMALL,
                 'large': HPC_DATA_FILEPATH+LARGE}
