package com.databricks.apps.logs.chapter2;

import com.databricks.apps.logs.ApacheAccessLog;
import com.databricks.apps.logs.LogAnalyzerRDD;
import com.databricks.apps.logs.LogStatistics;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.sql.SparkSession;

/**
 * LogAnalyzerBatchImport shows how to import data from a file.
 * 
 * Example command to run:
 * %  ${YOUR_SPARK_HOME}/bin/spark-submit
 *     --class "com.databricks.apps.logs.chapter2.LogAnalyzerBatchImport"
 *     --master spark://YOUR_SPARK_MASTER
 *     target/log-analyzer-2.0.jar
 *     ../../data/apache.access.log
 */
public class LogAnalyzerBatchImport {

    public static void main(String[] args) {
    SparkSession sparkSession = SparkSession
            .builder()
            .appName("Log Analyzer SQL")
            .getOrCreate();

    if (args.length == 0) {
      System.out.println("Must specify an access logs file.");
      System.exit(-1);
    }
    String logFile = args[0];
    JavaRDD<ApacheAccessLog> accessLogs = sparkSession
        .read()
        .textFile(logFile)
        .javaRDD()
        .map(ApacheAccessLog::parseFromLogLine);

    LogAnalyzerRDD logAnalyzerRDD = new LogAnalyzerRDD(sparkSession);
    LogStatistics logStatistics = logAnalyzerRDD.processRdd(accessLogs);
    logStatistics.printToStandardOut();

    sparkSession.stop();
  }
}
