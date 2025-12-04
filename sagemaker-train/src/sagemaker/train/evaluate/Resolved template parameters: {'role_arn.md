Resolved template parameters: {'role_arn':                       base_evaluator.py:757
                             'arn:aws:iam::634683118556:role/service-role/AmazonSageMaker-Exe                      
                             cutionRole-20251116T174807', 'mlflow_resource_arn':                                   
                             'arn:aws:sagemaker:us-east-1:634683118556:mlflow-app/app-DA25Q2S                      
                             35KHZ', 'mlflow_experiment_name': None, 'mlflow_run_name': None,                      
                             'model_package_group_arn':                                                            
                             'arn:aws:sagemaker:us-east-1:634683118556:model-package-group/tm                      
                             p-humanlike-llama32-rlaif', 'source_model_package_arn': None,                         
                             'base_model_arn':                                                                     
                             'arn:aws:sagemaker:us-east-1:aws:hub-content/SageMakerPublicHub/                      
                             Model/meta-textgeneration-llama-3-2-1b-instruct/1.25.0',                              
                             's3_output_path':                                                                     
                             's3://sagemaker-us-east-1-634683118556/tmp-humanlike-llama32-rla                      
                             if/eval', 'dataset_artifact_arn':                                                     
                             'arn:aws:sagemaker:us-east-1:634683118556:artifact/c3c6611071894                      
                             bad6a7f0925a729b02e', 'action_arn_prefix':                                            
                             'arn:aws:sagemaker:us-east-1:634683118556:action',                                    
                             'dataset_uri':                                                                        
                             'arn:aws:sagemaker:us-east-1:634683118556:hub-content/CKO4ACGI3U                      
                             RQBOO74C9JPLUMQNG02M2I4CIM9M931SQHE0625A30/DataSet/tmp-humanlike                      
                             -rlaif-eval/0.0.1', 'judge_model_id':                                                 
                             'anthropic.claude-3-5-sonnet-20240620-v1:0', 'llmaj_metrics':                         
                             '[]', 'custom_metrics_s3_path':                                                       
                             's3://sagemaker-us-east-1-634683118556/tmp-humanlike-llama32-rla                      
                             if/eval/evaluationinputs/eval-meta-1517aa3320251202-011237/custo                      
                             m-metrics.json', 'max_new_tokens': '8192', 'temperature': '0',                        
                             'top_k': '-1', 'top_p': '1.0', 'pipeline_name':                                       
                             'SagemakerModelEvaluationType2-llmaj', 'evaluate_base_model':                         
                             True}                                                                                 
                    INFO     Rendered pipeline definition:                                    base_evaluator.py:766
                             {                                                                                     
                               "Version": "2020-12-01",                                                            
                               "Metadata": {},                                                                     
                               "MlflowConfig": {                                                                   
                                 "MlflowResourceArn":                                                              
                             "arn:aws:sagemaker:us-east-1:634683118556:mlflow-app/app-DA25Q2S                      
                             35KHZ"                                                                                
                               },                                                                                  
                               "Parameters": [],                                                                   
                               "Steps": [                                                                          
                                 {                                                                                 
                                   "Name": "EvaluateBaseInferenceModel",                                           
                                   "Type": "Training",                                                             
                                   "Arguments": {                                                                  
                                     "TrainingJobName": "BaseInference",                                           
                                     "RoleArn":                                                                    
                             "arn:aws:iam::634683118556:role/service-role/AmazonSageMaker-Exe                      
                             cutionRole-20251116T174807",                                                          
                                     "ServerlessJobConfig": {                                                      
                                       "BaseModelArn":                                                             
                             "arn:aws:sagemaker:us-east-1:aws:hub-content/SageMakerPublicHub/                      
                             Model/meta-textgeneration-llama-3-2-1b-instruct/1.25.0",                              
                                       "AcceptEula": true,                                                         
                                       "JobType": "Evaluation",                                                    
                                       "EvaluationType": "BenchmarkEvaluation"                                     
                                     },                                                                            
                                     "StoppingCondition": {                                                        
                                       "MaxRuntimeInSeconds": 86400                                                
                                     },                                                                            
                                     "HyperParameters": {                                                          
                                       "name": "BaseInference",                                                    
                                       "task": "inference_only"                                                    
                                     },                                                                            
                                     "OutputDataConfig": {                                                         
                                       "S3OutputPath":                                                             
                             "s3://sagemaker-us-east-1-634683118556/tmp-humanlike-llama32-rla                      
                             if/eval",                                                                             
                                       "CompressionType": "NONE"                                                   
                                     },                                                                            
                                     "InputDataConfig": [                                                          
                                       {                                                                           
                                         "ChannelName": "train",                                                   
                                         "DataSource": {                                                           
                                           "DatasetSource": {                                                      
                                             "DatasetArn":                                                         
                             "arn:aws:sagemaker:us-east-1:634683118556:hub-content/CKO4ACGI3U                      
                             RQBOO74C9JPLUMQNG02M2I4CIM9M931SQHE0625A30/DataSet/tmp-humanlike                      
                             -rlaif-eval/0.0.1"                                                                    
                                           }                                                                       
                                         }                                                                         
                                       }                                                                           
                                     ]                                                                             
                                   }                                                                               
                                 },                                                                                
                                 {                                                                                 
                                   "Name": "EvaluateBaseModelMetrics",                                             
                                   "Type": "Training",                                                             
                                   "DependsOn": [                                                                  
                                     "EvaluateBaseInferenceModel"                                                  
                                   ],                                                                              
                                   "Arguments": {                                                                  
                                     "TrainingJobName": {                                                          
                                       "Std:Join": {                                                               
                                         "On": "-",                                                                
                                         "Values": [                                                               
                                           "base-llmaj-eval",                                                      
                                           {                                                                       
                                             "Get": "Execution.PipelineExecutionId"                                
                                           }                                                                       
                                         ]                                                                         
                                       }                                                                           
                                     },                                                                            
                                     "RoleArn":                                                                    
                             "arn:aws:iam::634683118556:role/service-role/AmazonSageMaker-Exe                      
                             cutionRole-20251116T174807",                                                          
                                     "ServerlessJobConfig": {                                                      
                                       "BaseModelArn":                                                             
                             "arn:aws:sagemaker:us-east-1:aws:hub-content/SageMakerPublicHub/                      
                             Model/meta-textgeneration-llama-3-2-1b-instruct/1.25.0",                              
                                       "AcceptEula": true,                                                         
                                       "JobType": "Evaluation",                                                    
                                       "EvaluationType": "LLMAJEvaluation"                                         
                                     },                                                                            
                                     "StoppingCondition": {                                                        
                                       "MaxRuntimeInSeconds": 86400                                                
                                     },                                                                            
                                     "HyperParameters": {                                                          
                                       "name": {                                                                   
                                         "Std:Join": {                                                             
                                           "On": "-",                                                              
                                           "Values": [                                                             
                                             "base-llmaj-eval",                                                    
                                             {                                                                     
                                               "Get": "Execution.PipelineExecutionId"                              
                                             }                                                                     
                                           ]                                                                       
                                         }                                                                         
                                       },                                                                          
                                       "judge_model_id":                                                           
                             "anthropic.claude-3-5-sonnet-20240620-v1:0",                                          
                                       "inference_data_s3_path": {                                                 
                                         "Std:Join": {                                                             
                                           "On": "",                                                               
                                           "Values": [                                                             
                                             {                                                                     
                                               "Get":                                                              
                             "Steps.EvaluateBaseInferenceModel.OutputDataConfig.S3OutputPath"                      
                                             },                                                                    
                                             "/",                                                                  
                                             {                                                                     
                                               "Get":                                                              
                             "Steps.EvaluateBaseInferenceModel.TrainingJobName"                                    
                                             },                                                                    
                                             "/output/output/",                                                    
                                             "BaseInference",                                                      
                                             "/eval_results/inference_output.jsonl"                                
                                           ]                                                                       
                                         }                                                                         
                                       },                                                                          
                                       "output_path":                                                              
                             "s3://sagemaker-us-east-1-634683118556/tmp-humanlike-llama32-rla                      
                             if/eval",                                                                             
                                       "llmaj_metrics": "[]",                                                      
                                       "custom_metrics_s3_path":                                                   
                             "s3://sagemaker-us-east-1-634683118556/tmp-humanlike-llama32-rla                      
                             if/eval/evaluationinputs/eval-meta-1517aa3320251202-011237/custo                      
                             m-metrics.json",                                                                      
                                       "max_new_tokens": "8192",                                                   
                                       "temperature": "0",                                                         
                                       "top_k": "-1",                                                              
                                       "top_p": "1.0"                                                              
                                     },                                                                            
                                     "OutputDataConfig": {                                                         
                                       "S3OutputPath":                                                             
                             "s3://sagemaker-us-east-1-634683118556/tmp-humanlike-llama32-rla                      
                             if/eval",                                                                             
                                       "CompressionType": "NONE"                                                   
                                     }                                                                             
                                   }                                                                               
                                 }                                                                                 
                               ]                                                                                   
                             } 