## Clone

Clone with submodules

```sh
$ git clone --recursive https://github.com/khadas/aml_npu_sdk.git
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
$ ./convert --model-name resnet18 --platform pytorch \
--model /home/yan/yan/Yan/models-zoo/pytorch/resnet18/resnet18.pt \
--input-size-list '3,224,224' \
--mean-values '103.94,116.78,123.68,58.82' \
--quantized-dtype asymmetric_affine \
--kboard VIM3 --print-level 1
```

library and nb file directoty: `outputs/resnet18`


