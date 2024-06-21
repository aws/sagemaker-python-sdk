from sagemaker.predictor import Predictor
from sagemaker import Session
from sagemaker.serializers import BaseSerializer, JSONSerializer
from sagemaker.deserializers import BaseDeserializer, JSONDeserializer


class DJLPredictor(Predictor):
    """A Predictor for inference against DJL Model Endpoints.

    This is able to serialize Python lists, dictionaries, and numpy arrays to
    multidimensional tensors for DJL inference.
    """

    def __init__(
        self,
        endpoint_name: str,
        sagemaker_session: Session = None,
        serializer: BaseSerializer = JSONSerializer(),
        deserializer: BaseDeserializer = JSONDeserializer(),
        component_name=None,
    ):
        """Initialize a ``DJLPredictor``

        Args:
            endpoint_name (str): The name of the endpoint to perform inference
                on.
            sagemaker_session (sagemaker.session.Session): Session object that
                manages interactions with Amazon SageMaker APIs and any other
                AWS services needed. If not specified, the estimator creates one
                using the default AWS configuration chain.
            serializer (sagemaker.serializers.BaseSerializer): Optional. Default
                serializes input data to json format.
            deserializer (sagemaker.deserializers.BaseDeserializer): Optional.
                Default parses the response from json format to dictionary.
            component_name (str): Optional. Name of the Amazon SageMaker inference
                component corresponding the predictor.
        """
        super(DJLPredictor, self).__init__(
            endpoint_name,
            sagemaker_session,
            serializer=serializer,
            deserializer=deserializer,
            component_name=component_name,
        )
