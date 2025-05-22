# Databricks notebook source
# MAGIC %md
# MAGIC ## SF crime data analysis and modeling
# MAGIC

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Project Introduction
# MAGIC In this notebook, the main goal is to use Spark SQL for big data analysis. \
# MAGIC Data used is open-source SF crime data from the following link: \
# MAGIC (https://data.sfgov.org/Public-Safety/Police-Department-Incident-Reports-Historical-2003/tmnf-yvry). 
# MAGIC
# MAGIC Project Member: Hao(Charlotte) Jin \
# MAGIC Project Finish Date: Apr 20th, 2025

# COMMAND ----------

# DBTITLE 1,Import package 
from csv import reader
from pyspark.sql import Row 
from pyspark.sql import SparkSession
from pyspark.sql.types import *
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import warnings
from pyspark.sql.functions import hour, date_format, to_date, month, year, concat,sum,count, round

import os
os.environ["PYSPARK_PYTHON"] = "python3"


# COMMAND ----------

# DBTITLE 1,Download data
# Download data from SF government website
import urllib.request
urllib.request.urlretrieve("https://data.sfgov.org/api/views/tmnf-yvry/rows.csv?accessType=DOWNLOAD", "/tmp/myxxxx.csv")
dbutils.fs.mv("file:/tmp/myxxxx.csv", "dbfs:/using_spark_analysis/spark_sf_crime_data/data/sf_05_21.csv")
display(dbutils.fs.ls("dbfs:/using_spark_analysis/spark_sf_crime_data/data/"))

# COMMAND ----------

data_path = "dbfs:/using_spark_analysis/spark_sf_crime_data/data/sf_05_21.csv"

# COMMAND ----------

# MAGIC %md
# MAGIC Load Data as TempView:

# COMMAND ----------

# DBTITLE 1,Load data as TempView

from pyspark.sql import SparkSession
spark = SparkSession \
    .builder \
    .appName("crime analysis") \
    .getOrCreate()
#.config("spark.some.config.option", "some-value") \

df_opt1 = spark.read.format("csv").option("header", "true").load(data_path)
display(df_opt1)
df_opt1.createOrReplaceTempView("sf_crime")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Conducting OLAP Tasks: 
# MAGIC #####Analysis on: the number of crimes for different category.

# COMMAND ----------

num_result = df_opt1.groupBy('category').count().orderBy('count', ascending=False)
#crimeCategory = spark.sql("SELECT category, COUNT(*) AS Count FROM sf_crime GROUP BY category ORDER BY Count DESC")
#crimes_pd_df = crimeCategory.toPandas()
display(num_result)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Analysis on: the number of crimes for different district
# MAGIC

# COMMAND ----------

district_count=spark.sql("""
                         select 
                            PdDistrict,
                            count(*) as num_of_crime
                          from sf_crime
                          group by PdDistrict
                          order by num_of_crime desc
                         """)
#display(district_count)

district_count_pd=district_count.filter(district_count.PdDistrict!="NA").toPandas()

district_count_pd=district_count_pd
plt.figure(figsize=(12,6))
plt.bar(district_count_pd['PdDistrict'],district_count_pd['num_of_crime'])
plt.xlabel('PD District')
plt.xticks(rotation=45)
plt.ylabel('Total Crimes')
plt.grid(axis='y',linestyle="--")
plt.title('Total Crimes by PD District')

# COMMAND ----------

# MAGIC %md
# MAGIC Based on the analysis, Southern, Mission, Northern are top three PD Districts having the most crimes. Richmond and Park, however, are relatively safer compared to other districts.

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Analysis on: the number of crime by month in 2015, 2016, 2017, 2018.

# COMMAND ----------

month_crime=spark.sql("""
                      select 
                         right(Date,4) as the_year,
                         left(Date,2) as the_month,
                         concat(right(Date,4),'-',left(Date,2)) as the_year_month,
                         count(*) as num_of_crime
                      from sf_crime
                      where right(Date,4) in ('2015','2016','2017','2018')
                      group by 1,2,3
                      order by the_year_month
                      """)
#display(month_crime)

month_crime_pd=month_crime.toPandas()

plt.figure(figsize=(15,6))
plt.bar(month_crime_pd['the_year_month'],month_crime_pd['num_of_crime'])
plt.xlabel('Year & Month')
plt.xticks(rotation=45)
plt.ylabel('Total Crimes')
plt.grid(axis='y',linestyle="--")
plt.title('Total Crimes by Year & Month period')

# COMMAND ----------

month_crime_g=month_crime.groupBy('the_month').agg(sum('num_of_crime').alias('num_of_crime'),
                                                   count('the_year').alias('count_of_year')).orderBy('the_month',ascending=True)
month_crime_g=month_crime_g.withColumn('avg_crime',round(month_crime_g.num_of_crime/month_crime_g.count_of_year,0))
#month_crime_g=month_crime_g.toDF('the_month','num_of_crime')

month_crime_g_pd=month_crime_g.toPandas()

plt.figure(figsize=(12,6))
plt.bar(month_crime_g_pd['the_month'],month_crime_g_pd['avg_crime'])
plt.xlabel('Month')
#plt.xticks(rotation=45)
plt.ylabel('Avg Crimes')
plt.grid(axis='y',linestyle="--")
plt.title('Average Crimes by Month in 2015~2018 period')

# COMMAND ----------

# MAGIC %md
# MAGIC Based on the analysis, looking at the period from 2015 to 2018, since we notice that there's only data till first half of 2018, directly adding the crimes up by month will skew months before June. So we use the average number of crimes for each year in that specific month as the metric here. 
# MAGIC
# MAGIC Based on the data, the crime number is usually higher in the summer and winter, especially October. SFPD might need to arrange more police force during this time. Feb and May are relatively a safer month. Looking at the data for each year, the number of crime in 2018 is significantly lower than previous years, which is a good sign for the city.

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Analysis on: the number of crime with respsect to hours of certian days 
# MAGIC (eg.2015/12/15, 2016/12/15, 2017/12/15)

# COMMAND ----------

hour_crime=spark.sql("""
                     select
                        left(time,2) as the_hour,
                        date,
                        count(*) as num_of_crime
                     from sf_crime
                     where Date in ('12/15/2015','12/15/2016','12/15/2017')
                     group by 1,2
                     order by date,the_hour
                     """)
#display(hour_crime)

# COMMAND ----------

hour_crime_g=hour_crime.groupBy('the_hour').agg(sum('num_of_crime').alias('num_of_crime')).orderBy('the_hour')

hour_crime_g_pd=hour_crime_g.toPandas()

plt.figure(figsize=(12,6))
plt.bar(hour_crime_g_pd['the_hour'],hour_crime_g_pd['num_of_crime'])
plt.xlabel('Hour in the day')
#plt.xticks(rotation=45)
plt.ylabel('Total Crimes')
plt.grid(axis='y',linestyle="--")
plt.title('Total Crimes by hour on 12/15 in 2015,2016,2017')

# COMMAND ----------

hour_crime_by_cate=spark.sql("""
                     select
                        left(time,2) as the_hour,
                        category,
                        count(*) as num_of_crime
                     from sf_crime
                     where Date in ('12/15/2015','12/15/2016','12/15/2017')
                     group by 1,2
                     order by the_hour
                     """)
display(hour_crime_by_cate)

# COMMAND ----------

# MAGIC %md
# MAGIC Based on the analysis, in a typical day, there usually are more crimes at noon (12:00pm) and early in the night, around 18:00 and 19:00. During these periods, larceny/theft is the most common crime. The number of crime is relatively low in the late night and in the early afternoon.
# MAGIC
# MAGIC As a tourist, it's suggested to go back to hotel or avoid walking on the street alone after 18:00. Also tourists should keep a closer eye on their belongings to avoid being robbed. 

# COMMAND ----------

# MAGIC %md 
# MAGIC ##### Analysis on: how to distribute the police force smartly
# MAGIC
# MAGIC (1) Step1: Find the top 3 most dangerous districts \
# MAGIC (2) Step2: Find the crime event w.r.t category and time (hour) in the top 3 most dangerous districts \
# MAGIC (3) Step3: Provide data-driven suggestions
# MAGIC

# COMMAND ----------

plt.figure(figsize=(12,6))
plt.bar(district_count_pd['PdDistrict'],district_count_pd['num_of_crime'])
plt.xlabel('PD District')
plt.xticks(rotation=45)
plt.ylabel('Total Crimes')
plt.grid(axis='y',linestyle="--")
plt.title('Total Crimes by PD District')

print('If considering whether a district is dangerous or not by looking at the total number of crimes, then the top 3 dangerous districts are: Southern, Mission, Northern. Graph:')

# COMMAND ----------

#Analysis for Southern District
hour_crime_cate=spark.sql("""
                     select
                        left(time,2) as the_hour,
                        PdDistrict,
                        category,
                        count(*) as num_of_crime
                     from sf_crime
                     where PdDistrict in ('SOUTHERN')
                    --where Date in ('12/15/2015','12/15/2016','12/15/2017')
                     group by 1,2,3
                     order by PdDistrict,the_hour
                     """)
display(hour_crime_cate)

# COMMAND ----------

#Analysis for Mission District
hour_crime_cate2=spark.sql("""
                     select
                        left(time,2) as the_hour,
                        PdDistrict,
                        category,
                        count(*) as num_of_crime
                     from sf_crime
                     where PdDistrict in ('MISSION')
                    --where Date in ('12/15/2015','12/15/2016','12/15/2017')
                     group by 1,2,3
                     order by PdDistrict,the_hour
                     """)
display(hour_crime_cate2)

# COMMAND ----------

#Analysis for Northern District
hour_crime_cate3=spark.sql("""
                     select
                        left(time,2) as the_hour,
                        PdDistrict,
                        category,
                        count(*) as num_of_crime
                     from sf_crime
                     where PdDistrict in ('NORTHERN')
                    --where Date in ('12/15/2015','12/15/2016','12/15/2017')
                     group by 1,2,3
                     order by PdDistrict,the_hour
                     """)
display(hour_crime_cate3)

# COMMAND ----------

hour_crime_cate.createOrReplaceTempView('southern')
hour_crime_cate2.createOrReplaceTempView('mission')
hour_crime_cate3.createOrReplaceTempView('northern')

compare=spark.sql("""
                  select *
                  from (select *, row_number() over(partition by the_hour order by num_of_crime desc) as the_rank from southern)
                  where the_rank<=5

                  union all

                  select *
                  from (select *, row_number() over(partition by the_hour order by num_of_crime desc) as the_rank from mission)
                  where the_rank<=5

                  union all

                  select *
                  from (select *, row_number() over(partition by the_hour order by num_of_crime desc) as the_rank from northern)
                  where the_rank<=5
                  """)

# COMMAND ----------

display(compare.filter(compare.PdDistrict=='SOUTHERN'))

# COMMAND ----------

display(compare.filter(compare.PdDistrict=='MISSION'))

# COMMAND ----------

display(compare.filter(compare.PdDistrict=='NORTHERN'))

# COMMAND ----------

# MAGIC %md
# MAGIC Based on the analysis, looking at the number of crimes for each district, we are able to identify that Southern, Mission and Northern districts are the top 3 most dangerous. By analyzing the number of crime by hours, we observe a similar trend for all three districts, with more crimes at noon (around 12:00pm) and early in the night (around 18:00).
# MAGIC
# MAGIC Southern district is difinitely the one with the highest crime volume at any hour during the day, which means we should arrange more police force to that district.
# MAGIC
# MAGIC For mission and northern districts, later in the night (around 22:00) the number of crimes aren't decreased as much as sourthern district, which means that we might need to arrange some extra police in case there's an emergency.
# MAGIC
# MAGIC In terms of the type of crime, for all these districts, theft and other offenses contribute the most. For mission and northern districts specificly, we observe an increasing volume of vehicle theft in the afternoon and in the night. As a result, we should arrange more patrol officers to watch out for vehicle theft.

# COMMAND ----------

# MAGIC %md 
# MAGIC ##### Analysis on: the percentage of resolution for different category of crime, and privide suggestions

# COMMAND ----------

resolution=spark.sql("""
                     with each_res as (
                     select
                        category,
                        resolution,
                        count(*) as count_num
                     from sf_crime
                     group by category,resolution
                     ),
                     
                     total_res as (
                     select
                        category,
                        count(*) as count_total
                     from sf_crime
                     group by category
                     )
                     --Note: when writing CTE, there should only be one 'With' keyword, and using ',' to seperate diff CTEs.

                     select
                        t1.*,
                        t2.count_total,
                        t1.count_num/t2.count_total as pct
                     from each_res t1
                     left join total_res t2 on t1.category=t2.category
                     """)
display(resolution)

# COMMAND ----------

resolution.createOrReplaceTempView('res_summary')

top_solution=spark.sql("""
                       select *
                       from (select *, row_number() over(partition by category order by pct desc) as the_rank from res_summary)
                       where the_rank<=3
                       """)
display(top_solution)

# COMMAND ----------

display(top_solution.filter((top_solution.resolution=="ARREST, BOOKED")|(top_solution.resolution=="ARREST, CITED"))\
            .groupBy('category')\
            .agg(sum('pct').alias('total_pct'))\
            .orderBy('total_pct',ascending=False)\
            .limit(5))

display(top_solution.filter(top_solution.resolution=="NONE")\
            .orderBy('pct',ascending=False)\
            .select('category','pct')\
            .limit(5))

#Use '\' at the end of each row, if you want to put the following codes to the next line
#Use '.select()' to select certain columns to display         

# COMMAND ----------

# MAGIC %md
# MAGIC Based on the analysis, there are lots of crimes ended up without a solution. Looking at the above table 2, which includes the ranking of all categories with 'None' as the resolution, about 90% of the 'Vehicle theft','Larceny/theft', 'Suspicious OCC','Vandalism' related crimes ended up without a resolution.
# MAGIC
# MAGIC However, for some serious crimes, like 'WARRANTS', 'DRIVING UNDER THE INFLUENCE', 'PROSTITUTION', 'DRUG/NARCOTIC', 'LIQUOR LAWS','LOITERING', and 'STOLEN PROPERTY', police can make the arrest successfully for about 90% of the overall crimes, which is a good trend.
# MAGIC
# MAGIC In terms of suggestion, it's gonna be helpful if police can be allocated more to smaller crimes, eg. theft, under the condition of not affecting the police performance on serious crimes. 

# COMMAND ----------

# MAGIC %md 
# MAGIC ##### Analysis on: the Central Market/Tenderloin Boundary in specific due to high traffic

# COMMAND ----------

td_b=((df_opt1['Central Market/Tenderloin Boundary 2 2']==1) 
      | (df_opt1['Central Market/Tenderloin Boundary Polygon - Updated 2 2']==1))
td_df=df_opt1.filter(td_b)

td_df.createOrReplaceTempView('td_crime')

td_analyze=spark.sql("""
                     select
                        right(date,4) as the_year,
                        category,
                        count(*) as num_of_crime                     
                     from td_crime
                     group by 1,2
                     order by num_of_crime desc
                     """)
display(td_analyze)

# COMMAND ----------

td_year=td_analyze.groupBy('the_year').agg(sum('num_of_crime').alias('num_of_crime')).orderBy('the_year',ascending=True)
td_year_pd=td_year.toPandas()

plt.figure(figsize=(24,12))
ax1=plt.subplot(2,2,1)
plt.bar(td_year_pd['the_year'],td_year_pd['num_of_crime'])
plt.xlabel('Year')
plt.ylabel('Total Crime')
plt.grid(axis='y',linestyle='--')
plt.title('Total Crimes in Tendorloin/Central Market boundary by Year')

ax2=plt.subplot(2,2,2)
td_year1=td_analyze.filter(td_analyze.category=="DRUG/NARCOTIC")
td_year_pd1=td_year1.toPandas()
plt.plot(td_year_pd1['the_year'],td_year_pd1['num_of_crime'])
plt.xlabel('Year')
plt.ylabel('Total Crime')
plt.grid(axis='y',linestyle='--')
plt.yticks(np.arange(0,7000,1000))
plt.title('Total Drug/Narcotic Crimes in Tendorloin/Central Market boundary by Year')


ax3=plt.subplot(2,2,3)
td_year3=td_analyze.filter(td_analyze.category=="LARCENY/THEFT")
td_year_pd3=td_year3.toPandas()
plt.plot(td_year_pd3['the_year'],td_year_pd3['num_of_crime'])
plt.xlabel('Year')
plt.ylabel('Total Crime')
plt.grid(axis='y',linestyle='--')
plt.yticks(np.arange(0,7000,1000))
plt.title('Total Larceny/Theft Crimes in Tendorloin/Central Market boundary by Year')


ax4=plt.subplot(2,2,4)
td_year4=td_analyze.filter(td_analyze.category=="ASSAULT")
td_year_pd4=td_year4.toPandas()
plt.plot(td_year_pd4['the_year'],td_year_pd4['num_of_crime'])
plt.xlabel('Year')
plt.ylabel('Total Crime')
plt.grid(axis='y',linestyle='--')
plt.yticks(np.arange(0,7000,1000))
plt.title('Total Assault Crimes in Tendorloin/Central Market boundary by Year')


# COMMAND ----------

# MAGIC %md
# MAGIC Based on the analysis, looking at the crime in central market/tenderloin boundary specificly, the overall number of crime decreased in recent years, starting 2013, making this area safer than before. 
# MAGIC
# MAGIC Looking at the crime categories, drug/narcotic is the major crime in this area. There are over 6000 crimes per year in 2008 and 2009. But thanks to the police, this number dropped significantly starting 2010, and it's already less than 1000 in 2018.
# MAGIC
# MAGIC Larceny/Theft is the second major crime in this area, and there's a significant increase in volume in recent years. It should be the next focus for SFPD. 
# MAGIC
# MAGIC Assault related crimes in this area is very steady in terms of the volume over the years, remaining to be around 2000 per year.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Conclusion
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Discover and suggestions
# MAGIC In order to provide a grading in terms of safety for citizen and tourists for San Francisco, we did a thorough OLAP analysis on the incident reports historical data of San Francisco police department. 
# MAGIC
# MAGIC Using Spark SQL to clean and analyze the database, and using Pyspark data visulization tools, we identified the top three most dangerous districts are Southern, Mission, and Northern districts, with huge number of larceny/theft, expecially at noon and in the evening. We suggest touriests and citizen should be more careful and pay attention to their belongings when having activities in these areas.
# MAGIC
# MAGIC In addition, we noticed that central market/tenderloin boundary is another area with huge number of crimes and drug related crimes are the major one there. Since it's very close to the tourist spots in San Francisco, we think tourists should be very careful when exploring in that downtown area. However, we do observed an significant decrease in the number of drug related crime thanks to the good work done by SFPD. Recently there are more and more theft related crime starting 2015, which should be the next target police should focus on.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Appendix

# COMMAND ----------

# DBTITLE 1,Notes on UDF
## helper function to transform the date, choose your way to do it. 
# refer: https://jaceklaskowski.gitbooks.io/mastering-spark-sql/spark-sql-functions-datetime.html

# Method 1: Using UDF within the system
# from pyspark.sql.functions import to_date, to_timestamp, hour
# df_opt1 = df_opt1.withColumn('Date', to_date(df_opt1.OccurredOn, "MM/dd/yy"))
# df_opt1 = df_opt1.withColumn('Time', to_timestamp(df_opt1.OccurredOn, "MM/dd/yy HH:mm"))
# df_opt1 = df_opt1.withColumn('Hour', hour(df_opt1['Time']))
# df_opt1 = df_opt1.withColumn("DayOfWeek", date_format(df_opt1.Date, "EEEE"))

## Method 2: 手工写udf 
#from pyspark.sql.functions import col, udf
#from pyspark.sql.functions import expr
#from pyspark.sql.functions import from_unixtime

#date_func =  udf (lambda x: datetime.strptime(x, '%m/%d/%Y'), DateType())
#month_func = udf (lambda x: datetime.strptime(x, '%m/%d/%Y').strftime('%Y/%m'), StringType())

#df = df_opt1.withColumn('month_year', month_func(col('Date')))\
#           .withColumn('Date_time', date_func(col('Date')))

## 方法3 手工在sql 里面
# select Date, substring(Date,7) as Year, substring(Date,1,2) as Month from sf_crime


## 方法4: 使用系统自带
# from pyspark.sql.functions import *
# df_update = df_opt1.withColumn("Date", to_date(col("Date"), "MM/dd/yyyy")) ##change datetype from string to date
# df_update.createOrReplaceTempView("sf_crime")
# crimeYearMonth = spark.sql("SELECT Year(Date) AS Year, Month(Date) AS Month, FROM sf_crime")