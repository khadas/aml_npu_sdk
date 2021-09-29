## Clone

Clone with submodules

```sh
$ git clone --recursive https://github.com/khadas/aml_npu_sdk.git
```

## Use Docker

Please install [Docker](https://docs.docker.com/) on your host PC.

### Build Docker image

```
$ docker pull numbqq/npu-sdk:latest
```

### Transcoding Model in Docker

```
$ docker run -it --name npu-sdk -v <model path>:/model --privileged  --cap-add SYS_ADMIN numbqq/npu-sdk
```

**Note: Replace `<model path>` to the path of model.**

Modify `conversion_scripts` for your model and transcodeing:

```
root@3cdfde2bcbc3:/acuity-toolkit/conversion_scripts# ./0_import_model.sh && ./1_quantize_model.sh && ./2_export_case_code.sh
```

Case code is in current directory: `nbg_unify_xxxxx`.
