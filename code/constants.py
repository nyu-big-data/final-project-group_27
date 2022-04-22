"""
A python file to import constants that point to data sources
"""
import getpass
netID = getpass.getuser() + "/"
HPC_DATA_FILEPATH = "hdfs:/user/" + netID
