val df = spark.read.parquet("s3a://logs/output/5-detect-predict")

df
  .select(
    "method", 
    "status", 
    "body_bytes_sent", 
    "http_referer", 
    "user_agent", 
    "dec_req_uri", 
    "fabstime", 
    "day_of_week", 
    "es_time", 
    "outlierScore",
    "anomaly_bool", 
    "domain_category"
  )
  .write
  .format("es")
  .mode("overwrite")
  .save("anom")


//http://localhost:5601/app/dashboards#/view/a26c06c7-2fcc-442c-b299-a89a7c46a324?_g=(refreshInterval:(pause:!t,value:60000),time:(from:'2024-01-14T10:15:59.349Z',to:now))&_a=(controlGroupInput:(chainingSystem:HIERARCHICAL,controlStyle:oneLine,id:control_group_dbe6c11d-c661-4c65-9413-5502d7b3cac3,ignoreParentSettings:(ignoreFilters:!f,ignoreQuery:!f,ignoreTimerange:!f,ignoreValidations:!f),panels:(b3788341-bd02-4689-95c2-4765a35bdf5a:(explicitInput:(dataViewId:'71bfa6f9-1bbf-4cc6-b434-b99115423301',enhancements:(),existsSelected:!t,fieldName:domain_category.keyword,grow:!t,id:b3788341-bd02-4689-95c2-4765a35bdf5a,searchTechnique:prefix,selectedOptions:!(),title:domain_category.keyword,width:medium),grow:!t,order:0,type:optionsListControl,width:medium)),showApplySelections:!f))
