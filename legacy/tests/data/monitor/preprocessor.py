import json
import random


# sample preprocess_handler (to be implemented by customer)
# This is a trivial example, where we demonstrate an echo preprocessor for json data
# for others though, we are generating random data (real customers would not do that obviously/hopefully)
def preprocess_handler(inference_record):
    event_data = inference_record.event_data
    input_data = {}
    output_data = {}

    # If the input data is JSON encoded, the following code just echoes it back
    if event_data.endpointInput.encoding == "JSON":
        input_data = json.loads(event_data.endpointInput.data)

    if event_data.endpointOutput.encoding == "JSON":
        output_data = json.loads(event_data.endpointOutput.data)

    # for non JSON data, this code just generates something random
    # real customers would read the event_data and transform it into a json
    if not input_data:
        input_data["feature0"] = random.uniform(0, 1)
        input_data["feature1"] = random.uniform(0, 1)

    if not output_data:
        output_data["prediction"] = random.uniform(0, 1)

    return {**input_data, **output_data}
