#!/usr/bin/env python3
import sys
from argparse import ArgumentParser
from acuitylib.console.completion import generate_completion
from acuitylib.console.custom_args import build_options,CustomArgs
from acuitylib.acuitymodel import CaffeLoader, AcuityModel, QuantizeType, TimVxExporter

def caffe_arguments():
    cmds = [
        CustomArgs({'cmd': '-m', 'help': '[model] Network model file.',
                    'type': 'file',
                    'required': True}),
        CustomArgs({'cmd': '-d', 'help': '[data] Network coefficient file.',
                    'type': 'file',
                    'required': True}),
        CustomArgs({'cmd': '-t', 'help': '[target] Target to Convert model.',
                    'required': True,
                    'choices': ['timvx']}),
    CustomArgs({'cmd': '-o',
                    'help': '[output] Network destination path.',
                    'required': True}),
    ]
    return cmds

def generate_arguments(with_help = True):
    def build_help_sub_commands(args):
        cmds = [CustomArgs({'cmd': '-a', 'choices': [arg.cmd for arg in args],
            'help': 'Print help message of an action.'})]
        return cmds
    args = [
        CustomArgs({'cmd': 'caffe',
            'help': 'covert caffe model.',
            'sub_cmds': caffe_arguments()}),
        ]

    if with_help is True:
        args += [CustomArgs({'cmd': 'help',
            'help': 'Print a synopsis and a list of commands.',
            'sub_cmds': build_help_sub_commands(args)})]
    return args

arguments = generate_arguments()
options = ArgumentParser(description='acuitygo commands.', prog='acuitygo')
build_options(options, arguments)

def parse_args():
    if len(sys.argv) == 2 and sys.argv[1] == 'completion':
        generate_completion('acuitygo', 'acuitygo_completion', arguments)
        sys.exit(0)
    if len(sys.argv) == 1:
        options.print_help()
        sys.exit(0)
    else:
        args = options.parse_args()
    return args

def convert_caffe(args):
    prototxt = args.m
    caffe_model = args.d
    export_path = args.o
    target = args.t
    importcaffe = CaffeLoader(model=prototxt, weights=caffe_model)
    net = importcaffe.load()
    model = AcuityModel(net)
    if target == 'timvx':
        if export_path == '.':
            export_path = "export_timvx/" + net.get_name()
        TimVxExporter(model).export(export_path)

def main(args):
    from acuitylib import Log as al
    al.reset_log_count()
    if args.which != 'help':
        al.i(args)
    ret = False
    if args.which == 'caffe':
        ret = convert_caffe(args)
    al.print_log_count()
    return ret

if __name__ == "__main__":
    args = parse_args()
    main(args)
