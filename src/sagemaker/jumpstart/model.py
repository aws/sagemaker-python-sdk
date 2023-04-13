from copy import deepcopy
from typing import Any, Optional
from sagemaker import environment_variables, image_uris, instance_types, model_uris, script_uris
from sagemaker.jumpstart.artifacts import _model_supports_prepacked_inference
from sagemaker.jumpstart.constants import JUMPSTART_DEFAULT_REGION_NAME
from sagemaker.model import Model
from sagemaker.session import Session


class JumpStartModel(Model):
    """JumpStartModel class.

    This class sets defaults based on the model id and version.
    """

    def __init__(
        self,
        model_id: str,
        model_version: Optional[str] = "*",
        instance_type: Optional[str] = None,
        region: Optional[str] = JUMPSTART_DEFAULT_REGION_NAME,
        kwargs_for_base_model_class: dict = {},
    ):
        self.model_id = model_id
        self.model_version = model_version
        self.kwargs_for_base_model_class = deepcopy(kwargs_for_base_model_class)

        self.instance_type = instance_type or instance_types.retrieve_default(
            region=region, model_id=model_id, model_version=model_version
        )

        self.kwargs_for_base_model_class = JumpStartModel._update_dict_if_key_not_present(
            self.kwargs_for_base_model_class,
            "image_uri",
            image_uris.retrieve(
                region=None,
                framework=None,
                image_scope="inference",
                model_id=model_id,
                model_version=model_version,
                instance_type=self.instance_type,
            ),
        )

        self.kwargs_for_base_model_class = JumpStartModel._update_dict_if_key_not_present(
            self.kwargs_for_base_model_class,
            "model_uri",
            model_uris.retrieve(
                script_scope="inference",
                model_id=model_id,
                model_version=model_version,
            ),
        )

        if not _model_supports_prepacked_inference(
            model_id=model_id, model_version=model_version, region=region
        ):
            self.kwargs_for_base_model_class = JumpStartModel._update_dict_if_key_not_present(
                self.kwargs_for_base_model_class,
                "script_uri",
                script_uris.retrieve(
                    script_scope="inference",
                    model_id=model_id,
                    model_version=model_version,
                ),
            )

        extra_env_vars = environment_variables.retrieve_default(
            region=region, model_id=model_id, model_version=model_version
        )

        curr_env_vars = self.kwargs_for_base_model_class.get("env", {})
        new_env_vars = deepcopy(curr_env_vars)

        for key, value in extra_env_vars:
            new_env_vars = JumpStartModel._update_dict_if_key_not_present(
                new_env_vars,
                key,
                value,
            )

        if new_env_vars == {}:
            new_env_vars = None

        self.kwargs_for_base_model_class["env"] = new_env_vars

        # model_kwargs_to_add = _retrieve_kwargs(model_id=model_id, model_version=model_version, region=region)
        model_kwargs_to_add = {}

        new_kwargs_for_base_model_class = deepcopy(self.kwargs_for_base_model_class)
        for key, value in model_kwargs_to_add:
            new_kwargs_for_base_model_class = JumpStartModel._update_dict_if_key_not_present(
                new_kwargs_for_base_model_class,
                key,
                value,
            )

        self.kwargs_for_base_model_class = new_kwargs_for_base_model_class

        self.kwargs_for_base_model_class["model_id"] = model_id
        self.kwargs_for_base_model_class["model_version"] = model_version

        # self.kwargs_for_base_model_class = JumpStartModel._update_dict_if_key_not_present(
        #     self.kwargs_for_base_model_class,
        #     "predictor_cls",
        #     JumpStartPredictor,
        # )

        super(Model, self).__init__(**self.kwargs_for_base_model_class)

    @staticmethod
    def _update_dict_if_key_not_present(
        dict_to_update: dict, key_to_add: Any, value_to_add: Any
    ) -> dict:
        if key_to_add not in dict_to_update:
            dict_to_update[key_to_add] = value_to_add

        return dict_to_update

    def deploy(self, **kwargs) -> callable[str, Session]:

        kwargs = JumpStartModel._update_dict_if_key_not_present(kwargs, "initial_instance_count", 1)
        kwargs = JumpStartModel._update_dict_if_key_not_present(
            kwargs, "instance_type", self.instance_type
        )

        return super(Model, self).deploy(**kwargs)
