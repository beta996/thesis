import mysql
from mysql.connector import Error
import sqlite3


connection = sqlite3.connect("db.name", check_same_thread=False)
cursor = connection.cursor()

# cursor.execute(
#     "CREATE TABLE historical_jobs ( id varchar(100) DEFAULT NULL, execution_time datetime NOT NULL, algorithm "
#     "varchar(45) DEFAULT NULL, config varchar(45) DEFAULT NULL,best_score float DEFAULT NULL,duration float DEFAULT "
#     "NULL,confusion_matrix varchar(50) DEFAULT NULL)")
#
# cursor.execute(
#     "CREATE TABLE jobs (  id varchar(45) DEFAULT NULL,  datasets varchar(100) DEFAULT NULL,  preprocessing_steps "
#     "varchar(200) DEFAULT NULL,  feature_extraction_method varchar(20) DEFAULT NULL,  feature_selection_percent int "
#     "DEFAULT NULL)")
#
# connection.commit()