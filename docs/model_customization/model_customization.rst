AI Model Customization Job Submission

SageMaker Python SDK V3 provides four specialized trainer classes for different model customization approaches:

**SFTTrainer (Supervised Fine-Tuning)**
  Traditional fine-tuning with labeled datasets for task-specific adaptation

**DPOTrainer (Direct Preference Optimization)**
  Fine-tune models using human preference data without reinforcement learning complexity

**RLAIFTrainer (Reinforcement Learning from AI Feedback)**
  Use AI-generated feedback to improve model behavior and alignment

**RLVRTrainer (Reinforcement Learning from Verifiable Rewards)**
  Fine-tune with verifiable reward signals for objective optimization

.. toctree::
   :maxdepth: 1
   :hidden:

    SFT Finetuning <../../v3-examples/model-customization-examples/sft_finetuning_example_notebook_pysdk_prod_v3>
    DPOTrainer Finetuning <../../v3-examples/model-customization-examples/dpo_trainer_example_notebook_v3_prod>
    RLAIF Finetuning <../../v3-examples/model-customization-examples/rlaif_finetuning_example_notebook_v3_prod>
    RLVR Finetuning <../../v3-examples/model-customization-examples/rlvr_finetuning_example_notebook_v3_prod>
