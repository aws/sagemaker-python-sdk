"""A dummy SageMaker job script testing local imports and requirements.txt installs"""

print("This is the print output from dummy_code_bundle_with_reqs/main_script.py")

print("Trying to import local module...")
import local_module

print("Trying to import module from requirements.txt...")
import stepfunctions

print("Done")
