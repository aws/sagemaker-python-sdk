
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
./aws/install

# TODO we should remove the boto model file once the Run API release
aws configure add-model --service-model file://resources/sagemaker-metrics-2022-09-30.normal.json --service-name sagemaker-metrics
aws configure add-model --service-model file://resources/sagemaker-2017-07-24.normal.json --service-name sagemaker

pip install resources/sagemaker-beta-1.0.tar.gz
python train_job_script_for_run_clz.py
