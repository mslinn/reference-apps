package com.databricks.apps.logs;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.SQLContext;

import scala.Tuple2;
import scala.Tuple4;

import java.util.List;

public class LogAnalyzerRDD {
  private final SQLContext sqlContext;

  public LogAnalyzerRDD(SQLContext sqlContext) {
    this.sqlContext = sqlContext;
  }

  public LogStatistics processRdd(JavaRDD<ApacheAccessLog> accessLogs) {
    // Spark SQL can imply a schema for a table if given a Java class with getters and setters.
    DataFrame sqlDataFrame = sqlContext.createDataFrame(accessLogs, ApacheAccessLog.class);
    sqlDataFrame.registerTempTable("logs");
    sqlContext.cacheTable("logs");

    LogStatistics stats = new LogStatistics();

    // Calculate statistics based on the content size.
    Tuple4<Long, Long, Long, Long> contentSizeStats =
        sqlContext.sql("SELECT SUM(contentSize), COUNT(*), MIN(contentSize), MAX(contentSize) FROM logs")
            .javaRDD()
            .map(row -> new Tuple4<>(row.getLong(0), row.getLong(1), row.getLong(2), row.getLong(3)))
            .first();
    stats.setContentSizeStats(contentSizeStats);

    // Compute Response Code to Count.
    List<Tuple2<Integer, Long>> responseCodeToCount = sqlContext
        .sql("SELECT responseCode, COUNT(*) FROM logs GROUP BY responseCode")
        .javaRDD()
        .mapToPair(row -> new Tuple2<>(row.getInt(0), row.getLong(1)))
        .take(1000);
    stats.setResponseCodeToCount(responseCodeToCount);

    // Any IPAddress that has accessed the server more than 10 times.
    List<String> ipAddresses = sqlContext
        .sql("SELECT ipAddress, COUNT(*) AS total FROM logs GROUP BY ipAddress HAVING total > 10")
        .javaRDD()
        .map(row -> row.getString(0))
        .take(100);  // Take only 100 in case this is a super large data set.
    stats.setIpAddresses(ipAddresses);

    // Top Endpoints.
    List<Tuple2<String, Long>> topEndpoints = sqlContext
        .sql("SELECT endpoint, COUNT(*) AS total FROM logs GROUP BY endpoint ORDER BY total DESC LIMIT 10")
        .javaRDD()
        .map(row -> new Tuple2<>(row.getString(0), row.getLong(1)))
        .collect();
    stats.setTopEndpoints(topEndpoints);

    return stats;
  }
}
