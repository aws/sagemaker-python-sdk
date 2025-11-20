import pytest
import pandas as pd
import numpy as np
import boto3
import joblib
import time
import os
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sagemaker.core.helper.session_helper import Session, get_execution_role
from sagemaker.core.clarify import (
    SageMakerClarifyProcessor,
    DataConfig,
    BiasConfig,
    SHAPConfig,
    _AnalysisConfigGenerator,
    ANALYSIS_CONFIG_SCHEMA_V1_0
)


@pytest.fixture
def sagemaker_session():
    return Session()


@pytest.fixture
def role():
    return get_execution_role()


@pytest.fixture
def test_data():
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        random_state=42
    )
    sensitive_feature = np.random.binomial(1, 0.4, size=X.shape[0])
    X = np.column_stack([X, sensitive_feature])
    feature_names = [f'feature_{i}' for i in range(10)] + ['gender']
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    return df


@pytest.fixture
def trained_model(test_data):
    X_train, X_test, y_train, y_test = train_test_split(
        test_data.drop('target', axis=1), test_data['target'], test_size=0.2, random_state=42
    )
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    return model, X_test, y_test


def test_clarify_e2e(sagemaker_session, role, test_data, trained_model):
    model, X_test, y_test = trained_model
    bucket = sagemaker_session.default_bucket()
    prefix = 'clarify-test'
    data_filename = 'clarify_bias_test_data.csv'
    model_filename = 'clarify_test_model.joblib'
    
    # Prepare test data
    test_df = X_test.copy()
    test_df['target'] = y_test
    test_df.to_csv(f'/tmp/{data_filename}', index=False)
    joblib.dump(model, f'/tmp/{model_filename}')
    
    # Upload to S3
    s3_client = boto3.client('s3')
    s3_client.upload_file(f'/tmp/{data_filename}', bucket, f'{prefix}/data/{data_filename}')
    s3_client.upload_file(f'/tmp/{model_filename}', bucket, f'{prefix}/model/{model_filename}')
    
    data_uri = f's3://{bucket}/{prefix}/data/{data_filename}'
    output_uri = f's3://{bucket}/{prefix}/output'
    
    # Configure Clarify
    data_config = DataConfig(
        s3_data_input_path=data_uri,
        s3_output_path=output_uri,
        label='target',
        headers=list(test_df.columns),
        dataset_type='text/csv'
    )
    
    bias_config = BiasConfig(
        label_values_or_threshold=[1],
        facet_name='gender',
        facet_values_or_threshold=[1]
    )
    
    shap_config = SHAPConfig(
        baseline=None,
        num_samples=10,
        agg_method='mean_abs'
    )
    
    # Create processor
    clarify_processor = SageMakerClarifyProcessor(
        role=role,
        instance_count=1,
        instance_type='ml.m5.large',
        sagemaker_session=sagemaker_session
    )
    
    # Run pre-training bias analysis
    clarify_processor.run_pre_training_bias(
        data_config=data_config,
        data_bias_config=bias_config,
        methods=['CI', 'DPL'],
        wait=False,
        logs=False
    )
    
    assert clarify_processor.latest_job is not None
    job_name = clarify_processor.latest_job.get_name()
    
    try:
        # Poll for job completion
        timeout = 600  # 10 minutes
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            response = sagemaker_session.sagemaker_client.describe_processing_job(
                ProcessingJobName=job_name
            )
            status = response['ProcessingJobStatus']
            
            if status == 'Completed':
                assert status == 'Completed'
                break
            elif status in ['Failed', 'Stopped']:
                pytest.fail(f"Processing job {status}: {response.get('FailureReason', 'Unknown')}")
            
            time.sleep(30)  # Wait 1 minute
        else:
            pytest.fail(f"Processing job timed out after {timeout} seconds")
    
    finally:
        # Cleanup S3 resources
        s3 = boto3.resource('s3')
        bucket_obj = s3.Bucket(bucket)
        bucket_obj.objects.filter(Prefix=f'{prefix}/').delete()
        
        # Cleanup local files
        for f in [f'/tmp/{data_filename}', f'/tmp/{model_filename}']:
            if os.path.exists(f):
                os.remove(f)


def test_bias_config_generation(sagemaker_session):
    bucket = sagemaker_session.default_bucket()
    data_uri = f"s3://{bucket}/test-clarify/data.csv"
    output_uri = f"s3://{bucket}/test-clarify/output"
    
    data_config = DataConfig(
        s3_data_input_path=data_uri,
        s3_output_path=output_uri,
        label='target',
        headers=['feature_0', 'gender', 'target'],
        dataset_type='text/csv'
    )
    
    bias_config = BiasConfig(
        label_values_or_threshold=[1],
        facet_name='gender',
        facet_values_or_threshold=[1]
    )
    
    bias_analysis_config = _AnalysisConfigGenerator.bias_pre_training(
        data_config=data_config,
        bias_config=bias_config,
        methods=['CI', 'DPL']
    )
    
    assert 'dataset_type' in bias_analysis_config
    assert 'label_values_or_threshold' in bias_analysis_config
    assert 'facet' in bias_analysis_config
    assert 'methods' in bias_analysis_config
    
    ANALYSIS_CONFIG_SCHEMA_V1_0.validate(bias_analysis_config)
