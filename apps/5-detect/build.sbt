name := "Anomaly detection"

version := "1.0"

scalaVersion := "2.12.18"

libraryDependencies += "org.apache.spark" %% "spark-sql" % "3.5.3"
libraryDependencies += "org.apache.spark" %% "spark-mllib" % "3.5.3"
libraryDependencies += "com.linkedin.isolation-forest" % "isolation-forest_3.5.0_2.12" % "3.0.6"

dependencyOverrides += "org.scala-lang.modules" %% "scala-xml" % "1.2.0"
