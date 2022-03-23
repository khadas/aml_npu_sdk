## Clone

Clone with submodules

```sh
$ git clone --recursive https://github.com/khadas/aml_npu_sdk.git
```

update submodules

```sh
$ git submodule init
$ git submodule update
```

c/c++ convert tool:

```
$ cd acuity-toolkit/demo
$ bash ./0_import_model.sh && bash ./1_quantize_model.sh && bash ./2_export_case_code.sh
```

Case code is in current directory: `nbg_unify_xxxxx`.

python convert tool:

```
$ cd acuity-toolkit/python
$ ./convert \
--model-name inception \
--platform tensorflow \
--model /path/to/inception_v3.pb \
--input-size-list '299,299,3' \
--inputs input \
--outputs InceptionV3/Predictions/Reshape_1 \
--mean-values '128 128 128 0.0078125' \
--quantized-dtype asymmetric_affine \
--source-files ./data/dataset/dataset0.txt \
--kboard VIM3 --print-level 1
```

library and nb file directoty: `outputs/inception`


