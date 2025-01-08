from argparse import ArgumentParser
import torch

def main():
    options = ArgumentParser(description='Torch PT File Input Output Pick Tool.')
    options.add_argument("-f",
                         required=True,
                         help="Torch pt file")
    options.add_argument("-mode",
                         required=False,
                         default='inputs_outputs',
                         type=str,
                         help="[graph] (print the whole graph) \n"
                              "[inputs_outputs] (print the default inputs outputs) \n"
                              "[file] (save graph in a txt file)")
    args = options.parse_args()
    pt_file = args.f
    show_mode = args.mode
    torch_model = torch.jit.load(pt_file)
    if show_mode == 'inputs_outputs':
        in_list = [i.debugName() for i in torch_model.graph.inputs()]
        in_list = in_list[1:]
        last_node = list(torch_model.graph.nodes())[-1]
        out_names = [o.debugName() for o in torch_model.graph.outputs()]
        # This special process is because Acuity Not support Sequence.
        # But if the ListConstruct work as Concat, this output should keep and the Convert also can handle it.
        if last_node.kind() in {"prim::ListConstruct", "prim::TupleConstruct"}:
            out_names.clear()
            out_names = [i.debugName() for i in last_node.inputs()]

        print("Input ID are ", in_list)
        print("Output ID are ", out_names)
    elif show_mode == 'graph':
        print(torch_model.graph)
    elif show_mode == 'file':
        fn = pt_file + '.graph.txt'
        print('Write Graph to ', fn)
        f = open(fn, 'w+')
        f.write(str(torch_model.graph))
        f.close()
    return





if __name__ == '__main__':
    main()