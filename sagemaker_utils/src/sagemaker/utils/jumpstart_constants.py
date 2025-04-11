import logging


ENV_VARIABLE_DISABLE_JUMPSTART_LOGGING = "DISABLE_JUMPSTART_LOGGING"
JUMPSTART_LOGGER = logging.getLogger("sagemaker.jumpstart")

# disable logging if env var is set
JUMPSTART_LOGGER.addHandler(
    type(
        "",
        (logging.StreamHandler,),
        {
            "emit": lambda self, *args, **kwargs: (
                logging.StreamHandler.emit(self, *args, **kwargs)
                if not os.environ.get(ENV_VARIABLE_DISABLE_JUMPSTART_LOGGING)
                else None
            )
        },
    )()
)