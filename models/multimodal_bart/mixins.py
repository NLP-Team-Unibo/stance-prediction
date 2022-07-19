import os
import itertools

from typing import Optional

import torch
from torch import nn
from torch.nn import Parameter

from transformers.generation_utils import logger, Iterable

#from transformers.modeling_bart import (
#    _reorder_buffer,
#    _make_linear_from_emb
#)

from transformers.modeling_utils import (
    hf_bucket_url,
    cached_path,
    TF2_WEIGHTS_NAME,
    WEIGHTS_NAME,
    TF_WEIGHTS_NAME,
    is_remote_url,
    PretrainedConfig
)

class FromPretrainedMixin:
    # This is based on torch.nn.module
    # The modifications are:
    # - if the parameter name is in config.partial_load, load only the part the state_dict saved
    @staticmethod
    def _load_from_state_dict(module, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs, partial_loads, config):
        r"""Copies parameters and buffers from :attr:`state_dict` into only
        this module, but not its descendants. This is called on every submodule
        in :meth:`~torch.nn.Module.load_state_dict`. Metadata saved for this
        module in input :attr:`state_dict` is provided as :attr:`local_metadata`.
        For state dicts without metadata, :attr:`local_metadata` is empty.
        Subclasses can achieve class-specific backward compatible loading using
        the version number at `local_metadata.get("version", None)`.
        .. note::
            :attr:`state_dict` is not the same object as the input
            :attr:`state_dict` to :meth:`~torch.nn.Module.load_state_dict`. So
            it can be modified.
        Arguments:
            state_dict (dict): a dict containing parameters and
                persistent buffers.
            prefix (str): the prefix for parameters and buffers used in this
                module
            local_metadata (dict): a dict containing the metadata for this module.
                See
            strict (bool): whether to strictly enforce that the keys in
                :attr:`state_dict` with :attr:`prefix` match the names of
                parameters and buffers in this module
            missing_keys (list of str): if ``strict=True``, add missing keys to
                this list
            unexpected_keys (list of str): if ``strict=True``, add unexpected
                keys to this list
            error_msgs (list of str): error messages should be added to this
                list, and will be reported together in
                :meth:`~torch.nn.Module.load_state_dict`
        """
        for hook in module._load_state_dict_pre_hooks.values():
            hook(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

        local_name_params = itertools.chain(module._parameters.items(), module._buffers.items())
        local_state = {k: v.data for k, v in local_name_params if v is not None}

        for name, param in local_state.items():
            key = prefix + name
            if key in state_dict:
                input_param = state_dict[key]

                # Backward compatibility: loading 1-dim tensor from 0.3.* to version 0.4+
                if len(param.shape) == 0 and len(input_param.shape) == 1:
                    input_param = input_param[0]

                if input_param.shape != param.shape:
                    # local shape should match the one in checkpoint
                    if key in config.partial_load:
                        partial_loads.append('partially loaded parameter {} ({} => {})'
                                             .format(key, input_param.shape, param.shape))
                    else:
                        error_msgs.append('size mismatch for {}: copying a param with shape {} from checkpoint, '
                                          'the shape in current model is {}.'
                                          .format(key, input_param.shape, param.shape))
                        continue

                if isinstance(input_param, Parameter):
                    # backwards compatibility for serialized parameters
                    input_param = input_param.data
                try:
                    if key in config.partial_load:
                        # copy only the parameters that is defined in input_param to param
                        param[tuple(map(slice, input_param.size()))].copy_(input_param)
                    else:
                        param.copy_(input_param)
                except Exception:
                    error_msgs.append('While copying the parameter named "{}", '
                                      'whose dimensions in the model are {} and '
                                      'whose dimensions in the checkpoint are {}.'
                                      .format(key, param.size(), input_param.size()))
            elif strict:
                missing_keys.append(key)

        if strict:
            for key in state_dict.keys():
                if key.startswith(prefix):
                    input_name = key[len(prefix):]
                    input_name = input_name.split('.', 1)[0]  # get the name of param/buffer/child
                    if input_name not in module._modules and input_name not in local_state:
                        unexpected_keys.append(key)

    # This is based on transformers.modeling_utils
    # The modifications are:
    # - changed module._load_from_state_dict to cls._load_from_state_dict
    # - warning on partial loads
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, error_on_mismatch=True, *model_args, **kwargs):
        r"""Instantiate a pretrained pytorch model from a pre-trained model configuration.
        The model is set in evaluation mode by default using model.eval() (Dropout modules are deactivated)
        To train the model, you should first set it back in training mode with model.train()
        The warning Weights from XXX not initialized from pretrained model means that the weights of XXX do not come
        pre-trained with the rest of the model.
        It is up to you to train those weights with a downstream fine-tuning task.
        The warning Weights from XXX not used in YYY means that the layer XXX is not used by YYY, therefore
        those weights are discarded.
        Parameters:
            pretrained_model_name_or_path: either:
              - a string with the shortcut name of a pre-trained model to load from cache or download,
                e.g.: bert-base-uncased.
              - a string with the identifier name of a pre-trained model that was user-uploaded to our S3,
                e.g.: dbmdz/bert-base-german-cased.
              - a path to a directory containing model weights saved using
                :func:~transformers.PreTrainedModel.save_pretrained, e.g.: ./my_model_directory/.
              - a path or url to a tensorflow index checkpoint file (e.g. ./tf_model/model.ckpt.index).
                In this case, from_tf should be set to True and a configuration object should be provided as config
                argument. This loading path is slower than converting the TensorFlow checkpoint in a PyTorch model
                using the provided conversion scripts and loading the PyTorch model afterwards.
              - None if you are both providing the configuration and state dictionary
                (resp. with keyword arguments config and state_dict)
            error_on_mismatch: (optional) boolean:
                Set to False to only warn the mismatch on loading weights, instead of error.
            model_args: (optional) Sequence of positional arguments:
                All remaning positional arguments will be passed to the underlying model's __init__ method
            config: (optional) one of:
                - an instance of a class derived from :class:~transformers.PretrainedConfig, or
                - a string valid as input to :func:~transformers.PretrainedConfig.from_pretrained()
                Configuration for the model to use instead of an automatically loaded configuation.
                Configuration can be automatically loaded when:
                    - the model is a model provided by the library (loaded with the shortcut-name string of
                      a pretrained model), or
                    - the model was saved using :func:~transformers.PreTrainedModel.save_pretrained and is
                      reloaded by suppling the save directory.
                    - the model is loaded by suppling a local directory as pretrained_model_name_or_path and
                      a configuration JSON file named config.json is found in the directory.
            state_dict: (optional) dict:
                an optional state dictionnary for the model to use instead of a state dictionary loaded from
                saved weights file. This option can be used if you want to create a model from a pretrained
                configuration but load your own weights. In this case though, you should check if using
                :func:~transformers.PreTrainedModel.save_pretrained and
                :func:~transformers.PreTrainedModel.from_pretrained is not a simpler option.
            cache_dir: (optional) string:
                Path to a directory in which a downloaded pre-trained model
                configuration should be cached if the standard cache should not be used.
            force_download: (optional) boolean, default False:
                Force to (re-)download the model weights and configuration files and override the cached versions
                if they exists.
            resume_download: (optional) boolean, default False:
                Do not delete incompletely recieved file. Attempt to resume the download if such a file exists.
            proxies: (optional) dict, default None:
                A dictionary of proxy servers to use by protocol or endpoint,
                e.g.: {'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}.
                The proxies are used on each request.
            output_loading_info: (optional) boolean:
                Set to True to also return a dictionnary containing missing keys, unexpected keys and error messages.
            kwargs: (optional) Remaining dictionary of keyword arguments:
                Can be used to update the configuration object (after it being loaded) and initiate the model.
                (e.g. output_attention=True). Behave differently depending on whether a config is provided
                or automatically loaded:
                - If a configuration is provided with config, **kwargs will be directly passed to the underlying
                model's __init__ method (we assume all relevant updates to the configuration have already been done)
                - If a configuration is not provided, kwargs will be first passed to the configuration class
                initialization function (:func:~transformers.PretrainedConfig.from_pretrained).
                Each key of kwargs that corresponds to a configuration attribute will be used to override said
                attribute with the supplied kwargs value. Remaining keys that do not correspond to any
                configuration attribute will be passed to the underlying model's __init__ function.
        """
        config = kwargs.pop("config", None)
        state_dict = kwargs.pop("state_dict", None)
        cache_dir = kwargs.pop("cache_dir", None)
        from_tf = kwargs.pop("from_tf", False)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        output_loading_info = kwargs.pop("output_loading_info", False)
        local_files_only = kwargs.pop("local_files_only", False)
        #use_cdn = kwargs.pop("use_cdn", True)

        # Load config if we don't provide a configuration
        if not isinstance(config, PretrainedConfig):
            config_path = config if config is not None else pretrained_model_name_or_path
            config, model_kwargs = cls.config_class.from_pretrained(
                config_path,
                *model_args,
                cache_dir=cache_dir,
                return_unused_kwargs=True,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                local_files_only=local_files_only,
                **kwargs,
            )
        else:
            model_kwargs = kwargs

        # Load model
        if pretrained_model_name_or_path is not None:
            if os.path.isdir(pretrained_model_name_or_path):
                if from_tf and os.path.isfile(os.path.join(pretrained_model_name_or_path, TF_WEIGHTS_NAME + ".index")):
                    # Load from a TF 1.0 checkpoint
                    archive_file = os.path.join(pretrained_model_name_or_path, TF_WEIGHTS_NAME + ".index")
                elif from_tf and os.path.isfile(os.path.join(pretrained_model_name_or_path, TF2_WEIGHTS_NAME)):
                    # Load from a TF 2.0 checkpoint
                    archive_file = os.path.join(pretrained_model_name_or_path, TF2_WEIGHTS_NAME)
                elif os.path.isfile(os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)):
                    # Load from a PyTorch checkpoint
                    archive_file = os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)
                else:
                    raise EnvironmentError(
                        "Error no file named {} found in directory {} or from_tf set to False".format(
                            [WEIGHTS_NAME, TF2_WEIGHTS_NAME, TF_WEIGHTS_NAME + ".index"],
                            pretrained_model_name_or_path,
                        )
                    )
            elif os.path.isfile(pretrained_model_name_or_path) or is_remote_url(pretrained_model_name_or_path):
                archive_file = pretrained_model_name_or_path
            elif os.path.isfile(pretrained_model_name_or_path + ".index"):
                assert (
                    from_tf
                ), "We found a TensorFlow checkpoint at {}, please set " \
                   "from_tf to True to load from this checkpoint".format(
                    pretrained_model_name_or_path + ".index"
                )
                archive_file = pretrained_model_name_or_path + ".index"
            else:
                archive_file = hf_bucket_url(
                    pretrained_model_name_or_path,
                    filename=(TF2_WEIGHTS_NAME if from_tf else WEIGHTS_NAME),
                    #use_cdn=use_cdn,
                )

            try:
                # Load from URL or cache if already cached
                resolved_archive_file = cached_path(
                    archive_file,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    local_files_only=local_files_only,
                )
                if resolved_archive_file is None:
                    raise EnvironmentError
            except EnvironmentError:
                msg = (
                    f"Can't load weights for '{pretrained_model_name_or_path}'. Make sure that:\n\n"
                    f"- '{pretrained_model_name_or_path}' is a correct model identifier listed on "
                    f"'https://huggingface.co/models'\n\n"
                    f"- or '{pretrained_model_name_or_path}' is the correct path to a directory "
                    f"containing a file named one of {WEIGHTS_NAME}, {TF2_WEIGHTS_NAME}, {TF_WEIGHTS_NAME}.\n\n"
                )
                raise EnvironmentError(msg)

            if resolved_archive_file == archive_file:
                logger.info("loading weights file {}".format(archive_file))
            else:
                logger.info("loading weights file {} from cache at {}".format(archive_file, resolved_archive_file))
        else:
            resolved_archive_file = None

        # Instantiate model.
        model = cls(config, *model_args, **model_kwargs)

        if state_dict is None and not from_tf:
            try:
                state_dict = torch.load(resolved_archive_file, map_location="cpu")
            except Exception:
                raise OSError(
                    "Unable to load weights from pytorch checkpoint file. "
                    "If you tried to load a PyTorch model from a TF 2.0 checkpoint, please set from_tf=True. "
                )

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        partial_loads = []

        if from_tf:
            if resolved_archive_file.endswith(".index"):
                # Load from a TensorFlow 1.X checkpoint - provided by original authors
                model = cls.load_tf_weights(model, config, resolved_archive_file[:-6])  # Remove the '.index'
            else:
                # Load from our TensorFlow 2.0 checkpoints
                try:
                    from transformers import load_tf2_checkpoint_in_pytorch_model

                    model = load_tf2_checkpoint_in_pytorch_model(model, resolved_archive_file, allow_missing_keys=True)
                except ImportError:
                    logger.error(
                        "Loading a TensorFlow model in PyTorch, requires both PyTorch and "
                        "TensorFlow to be installed. Please see "
                        "https://pytorch.org/ and https://www.tensorflow.org/install/ for installation instructions."
                    )
                    raise
        else:
            # Convert old format to new format if needed from a PyTorch state_dict
            old_keys = []
            new_keys = []
            for key in state_dict.keys():
                new_key = None
                if "gamma" in key:
                    new_key = key.replace("gamma", "weight")
                if "beta" in key:
                    new_key = key.replace("beta", "bias")
                if new_key:
                    old_keys.append(key)
                    new_keys.append(new_key)
            for old_key, new_key in zip(old_keys, new_keys):
                state_dict[new_key] = state_dict.pop(old_key)

            # copy state_dict so _load_from_state_dict can modify it
            metadata = getattr(state_dict, "_metadata", None)
            state_dict = state_dict.copy()
            if metadata is not None:
                state_dict._metadata = metadata

            # PyTorch's _load_from_state_dict does not copy parameters in a module's descendants
            # so we need to apply the function recursively.
            def load(module: nn.Module, prefix=""):
                local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
                cls._load_from_state_dict(
                    module=module,
                    state_dict=state_dict,
                    prefix=prefix,
                    local_metadata=local_metadata,
                    strict=True,
                    missing_keys=missing_keys,
                    unexpected_keys=unexpected_keys,
                    error_msgs=error_msgs,
                    partial_loads=partial_loads,
                    config=config
                )
                for name, child in module._modules.items():
                    if child is not None:
                        load(child, prefix + name + ".")

            # Make sure we are able to load base models as well as derived models (with heads)
            start_prefix = ""
            model_to_load = model
            has_prefix_module = any(s.startswith(cls.base_model_prefix) for s in state_dict.keys())
            if not hasattr(model, cls.base_model_prefix) and has_prefix_module:
                start_prefix = cls.base_model_prefix + "."
            if hasattr(model, cls.base_model_prefix) and not has_prefix_module:
                model_to_load = getattr(model, cls.base_model_prefix)

            load(model_to_load, prefix=start_prefix)

            if model.__class__.__name__ != model_to_load.__class__.__name__:
                base_model_state_dict = model_to_load.state_dict().keys()
                head_model_state_dict_without_base_prefix = [
                    key.split(cls.base_model_prefix + ".")[-1] for key in model.state_dict().keys()
                ]

                missing_keys.extend(head_model_state_dict_without_base_prefix - base_model_state_dict)

            if len(unexpected_keys) > 0:
                logger.warning(
                    f"Some weights of the model checkpoint at {pretrained_model_name_or_path} were not used when "
                    f"initializing {model.__class__.__name__}: {unexpected_keys}\n"
                    f"- This IS expected if you are initializing {model.__class__.__name__} from the checkpoint "
                    f"of a model trained on another task "
                    f"or with another architecture (e.g. initializing a BertForSequenceClassification model from "
                    f"a BertForPretraining model).\n"
                    f"- This IS NOT expected if you are initializing {model.__class__.__name__} from the checkpoint "
                    f"of a model that you expect "
                    f"to be exactly identical (initializing a BertForSequenceClassification model from "
                    f"a BertForSequenceClassification model)."
                )
            else:
                logger.info(f"All model checkpoint weights were used when initializing {model.__class__.__name__}.\n")
            if len(missing_keys) > 0:
                logger.warning(
                    f"Some weights of {model.__class__.__name__} were not initialized from the "
                    f"model checkpoint at {pretrained_model_name_or_path} "
                    f"and are newly initialized: {missing_keys}\n"
                    f"You should probably TRAIN this model on a down-stream task to be able "
                    f"to use it for predictions and inference."
                )
            else:
                logger.info(
                    f"All the weights of {model.__class__.__name__} were initialized from "
                    f"the model checkpoint at {pretrained_model_name_or_path}.\n"
                    f"If your task is similar to the task the model of the ckeckpoint was trained on, "
                    f"you can already use {model.__class__.__name__} for predictions without further training."
                )
            if len(error_msgs) > 0:
                RuntimeError("Error(s) in loading state_dict for {}:\n\t{}".format(
                    model.__class__.__name__, "\n\t".join(error_msgs)
                ))
            if len(partial_loads) > 0:
                logger.info("Partial loads while loading state_dict for {}:\n\t{}".format(
                    model.__class__.__name__, "\n\t".join(partial_loads)
                ))
        model.tie_weights()  # make sure token embedding weights are still tied if needed

        # Set model in evaluation mode to deactivate DropOut modules by default
        model.eval()

        if output_loading_info:
            loading_info = {
                "missing_keys": missing_keys,
                "unexpected_keys": unexpected_keys,
                "error_msgs": error_msgs,
            }
            return model, loading_info

        if hasattr(config, "xla_device") and config.xla_device:
            import torch_xla.core.xla_model as xm

            model = xm.send_cpu_data_to_device(model, xm.xla_device())
            model.to(xm.xla_device())

        return model