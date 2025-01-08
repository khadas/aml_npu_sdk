import setuptools

import sys, os, shutil, time
import copy
from Cython.Build import cythonize
from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
from pathlib import Path
from multiprocessing import Process
import ruamel.yaml as yaml
from collections import namedtuple, OrderedDict

CommandLineMappingDef = namedtuple('acuitycommandlinetoolsdef', ['dst_folder', 'copy_options'])
command_line_name_mappings = OrderedDict()
command_line_name_mappings['./acuitynet/'] = \
    CommandLineMappingDef('acuity_command_line_tools/', "normal")
command_line_name_mappings['./acuitynet/client/'] = \
    CommandLineMappingDef('acuity_command_line_tools/client', 'normal')
command_line_name_mappings['./acuitynet/acuitylib/app/exporter/ovxlib_case/vxcode/'] = \
    CommandLineMappingDef('acuity_command_line_tools/vxcode', 'recursive')
command_line_name_mappings['./acuitynet/tools/'] = \
    CommandLineMappingDef('acuity_command_line_tools/tools', "recursive")
command_line_name_mappings['./acuitynet/acuitylib/converter/tensorflow/tfruler/'] = \
    CommandLineMappingDef('acuity_command_line_tools/tfruler', "recursive")
command_line_name_mappings['./acuitynet/acuitylib/converter/onnx/onnxruler/'] = \
    CommandLineMappingDef('acuity_command_line_tools/onnxruler', "recursive")
command_line_name_mappings['./acuitynet/acuitylib/converter/pytorch/pytorchirruler/'] = \
    CommandLineMappingDef('acuity_command_line_tools/pytorchirruler', "recursive")

# command_line_name_mappings2 = OrderedDict()
# command_line_name_mappings2['../acuitynet/'] = \
#     CommandLineMappingDef('acuity_command_line_tools/', "normal")
# command_line_name_mappings2['../acuitynet/client/'] = \
#     CommandLineMappingDef('acuity_command_line_tools/client', 'normal')
# command_line_name_mappings2['../acuitynet/vxcode/'] = \
#     CommandLineMappingDef('acuity_command_line_tools/vxcode', 'recursive')
# command_line_name_mappings2['../acuitynet/tools/'] = \
#     CommandLineMappingDef('acuity_command_line_tools/tools', "recursive")
# command_line_name_mappings2['../acuitynet/acuitylib/converter/tfruler/'] = \
#     CommandLineMappingDef('acuity_command_line_tools/tfruler', "recursive")
# command_line_name_mappings2['../acuitynet/acuitylib/converter/onnx/onnxruler/'] = \
#     CommandLineMappingDef('acuity_command_line_tools/onnx/onnxruler', "recursive")
# command_line_name_mappings2['../acuitynet/acuitylib/converter/pytorch/pytorchirruler/'] = \
#     CommandLineMappingDef('acuity_command_line_tools/pytorchirruler', "recursive")

def get_folder_tree_yaml_obj(root_folder, json_file_name, yaml_file_name=''):

    json_file_path = root_folder + os.path.sep + json_file_name
    yaml_file_path = root_folder + os.path.sep + yaml_file_name

    # To get a clean wksp, remove temp files if exists
    if os.path.exists(json_file_path):
        os.remove(json_file_path)
    if os.path.exists(yaml_file_path):
        os.remove(yaml_file_path)

    print("Generate directory structure json...")
    tree_command = "tree {} -J -o {} --noreport -I '{}|__pycache__|build|dist|acuity_src_*|acuity_bin_*|log'"\
        .format(root_folder, json_file_name, json_file_name)
    os.system(tree_command)
    print("Generate directory structure json file, Done.")

    print("Generate directory structure yaml object...")
    json_f = open(json_file_path, 'r')
    tree_yaml_obj = yaml.safe_load(json_f.read())
    print("Generate directory structure yaml object, Done.")

    if yaml_file_name != '':
        yaml_f = open(yaml_file_path,'w')
        yaml.safe_dump(tree_yaml_obj,yaml_f)
        print("Generate directory structure yaml file, Done.")

    return tree_yaml_obj

ManifestParseResultDef = namedtuple('manifestparsemapping', ['type', 'pack', 'forcetype'])
def yaml_walker_get_pack_files(pack_edition, path, obj, out_dict, root_folder, replace):
    for item in obj:
        out = path + item['name']
        if replace:
            #replace the root folder name to . to align with manifest
            out = out.replace(root_folder, '.', 1)
        if item['type'] != 'directory':
            if pack_edition in item['pack']:
                if 'forcetype' in item:
                    forcetype = item['forcetype']
                else:
                    forcetype = ''
                out_dict[out] = ManifestParseResultDef(item['type'], item['pack'], forcetype)
        #print(out,flush=True)
        if item['type'] == 'directory':
            yaml_walker_get_pack_files(pack_edition, path + item['name'] + os.path.sep, item['contents'],
                                       out_dict, root_folder, replace)

def yaml_walker(pack_edition, path, obj, out_tuple_list, mappings):
    file_list = []
    if path == '':
        current_dir = '.'
    else:
        current_dir = path

    for item in obj:

        out = path + item['name']

        if path in mappings.keys():
            if item['type'] != 'directory':
                if pack_edition in item['pack']:
                    file_list.append(out.replace('./acuitynet/', './', 1))
            else:
                #item is a folder
                #if  its parent folder is recursive
                if mappings[path].copy_options == 'recursive':
                    #print('append subfolder -> ', out + os.path.sep)
                    #add the folder to mappings and mark its copy_option as recursive
                    mappings[out + os.path.sep] = \
                        CommandLineMappingDef(mappings[path].dst_folder + os.path.sep + item['name'], "recursive")

        if item['type'] == 'directory':
            yaml_walker(pack_edition, path + item['name'] + os.path.sep, item['contents'],
                        out_tuple_list, mappings)

    if len(file_list) > 0:
        out_tuple_list.append((mappings[current_dir].dst_folder, file_list))

def yaml_walker_and_copy(pack_edition, path, diliverable_folder, obj, out_tuple_list):
    for item in obj:

        out = path + item['name']

        if item['type'] != 'directory':
            if pack_edition in item['pack']:
                src_file_path = out.replace('.', '..', 1)
                dst_file_folder = diliverable_folder + path.replace('.', '', 1)
                if not os.path.exists(dst_file_folder):
                    os.makedirs(dst_file_folder)
                print("Copy {} to {}".format(src_file_path, dst_file_folder))
                shutil.copy(src_file_path, dst_file_folder)
        else:
            yaml_walker_and_copy(pack_edition, path + item['name'] + os.path.sep, diliverable_folder, item['contents'],
                        out_tuple_list)

def folder_walker(path, out_tuple_list, mappings):
    if path in mappings.keys():

        file_list = []
        for name in os.listdir(path):
            pathname = os.path.join(path,name)
            if not os.path.isdir(pathname):
                file_list.append(pathname)
            else:
                if mappings[path].copy_options == 'recursive':
                    mappings[pathname + os.path.sep] =\
                        CommandLineMappingDef(mappings[path].dst_folder + os.path.sep + name, "recursive")

            if mappings[path].copy_options == 'recursive' and os.path.isdir(pathname):
                folder_walker(pathname + os.path.sep, out_tuple_list, mappings)

        if len(file_list) > 0:
            out_tuple_list.append((mappings[path].dst_folder, file_list))


def main():
    build_package_type = sys.argv.pop(1)
    pack_edition = sys.argv.pop(1)
    verify_manifest = sys.argv.pop(1)

    with open('README.md', 'r') as f:
        long_description = f.read()

    with open('VERSION', 'r') as f:
        version = f.read().strip()

    REQUIRED_PACKAGES = [
        'lmdb == 0.93',
        'networkx >= 1.11',
        'onnx == 1.8.0',
        'flatbuffers == 1.10',
        'dill == 0.2.8.2',
        'ruamel.yaml == 0.15.81',
        'ply == 3.11',
        'torch == 1.5.1'
    ]
    REQUIRED_PACKAGES_CPU_VERSION = ['tensorflow == 2.3.0']
    REQUIRED_PACKAGES_GPU_VERSION = ['tensorflow-gpu == 2.3.0']

    REQUIRED_PACKAGES_CPU_VERSION.extend(REQUIRED_PACKAGES)
    REQUIRED_PACKAGES_GPU_VERSION.extend(REQUIRED_PACKAGES)

    CODE_TEMPLATE = {}

    BUILDTYPE = ''
    if build_package_type == 'build_test':
        BUILDTYPE = 'test'
    elif build_package_type == 'build_dev':
        BUILDTYPE = 'dev'
    elif build_package_type == 'build_rel':
        BUILDTYPE = 'rel'
    elif build_package_type == 'build_whl_src':
        BUILDTYPE = 'whl_src'
    elif build_package_type == 'build_whl_bin':
        BUILDTYPE = 'whl'
    elif build_package_type == 'build_binary_bin':
        BUILDTYPE = 'binary'
    else:
        print("Error: Unknown build package type {}.".format(build_package_type))
    VERSION = ''
    with open('VERSION', 'r') as f:
        VERSION = f.read().strip()
    DELIVERABLE = '..{}acuity-toolkit-{}-{}'.format(os.path.sep, BUILDTYPE, VERSION)
    DELIVERABLE_BIN = DELIVERABLE + os.path.sep + 'bin'

    manifest_yaml_obj = yaml.safe_load(open('../manifest.yml','r').read())

    ACUITY_COMMAND_LINE_TOOLS = []
    if verify_manifest == 'true' and BUILDTYPE in ['dev', 'rel']:
        yaml_walker_and_copy(pack_edition, '', DELIVERABLE, manifest_yaml_obj, [])
        return 0
    yaml_walker(pack_edition, '', manifest_yaml_obj,
                ACUITY_COMMAND_LINE_TOOLS, command_line_name_mappings)
    # else:
    #     need_walk_folders = copy.deepcopy(list(command_line_name_mappings2.keys()))
    #     for need_walk_folder in need_walk_folders:
    #         folder_walker(need_walk_folder, ACUITY_COMMAND_LINE_TOOLS, command_line_name_mappings2)

    def copy_command_line_tools(command_line_tools_dict):
        print("Copy command line tools relative files...")
        #clean deliverable folder
        os.system("rm -rf {}".format(DELIVERABLE))

        #copy required files(command line tools, vxcode, client, pb2.py etc.) to the DELIVERABLE_BIN
        #this step is prepare files for build_linux.sh to pack
        for item in command_line_tools_dict:
            dst_folder_in_deliverable_bin = DELIVERABLE_BIN + item[0].replace('acuity_command_line_tools/','/')
            if not os.path.exists(dst_folder_in_deliverable_bin):
                os.makedirs(dst_folder_in_deliverable_bin)
            for file_path in item[1]:
                print("Copy {} to {}".format(file_path, dst_folder_in_deliverable_bin))
                shutil.copy(file_path, dst_folder_in_deliverable_bin)

        print("Copy command line tools relative files done.")

    #CAUTION!!! for binary package we only need copy command line tools
    if BUILDTYPE == 'binary':
        copy_command_line_tools(ACUITY_COMMAND_LINE_TOOLS)
        return 0

    pack_start_time = time.time()
    rootdir = os.path.abspath('.')
    parentpath = "acuitylib"
    setupfile = os.path.join(os.path.abspath('.'), __file__)
    build_dir = "build"
    build_tmp_dir = build_dir + "/temp"
    client_module_folder = "client"

    class MyBuildExt(build_ext):
        manifest = None
        build_package_type = None

        @classmethod
        def set_manifest(self, manifest = None):
            MyBuildExt.manifest = manifest

        @classmethod
        def set_build_package_type(self, package_type = None):
            MyBuildExt.build_package_type = package_type

        def run(self):
            build_ext.run(self)

            build_dir = Path(self.build_lib)
            root_dir = Path(__file__).parent

            target_dir = build_dir if not self.inplace else root_dir  # CAUTION self.inplace change
            self.copy_file_recusive(rootdir, parentpath, '', str(target_dir))

        def copy_file(self, source_dir, file_path, destination_dir):
            if not (source_dir / file_path).exists():
                return
            shutil.copyfile(str(source_dir / file_path), str(destination_dir / file_path))

        def copy_file_recusive(self, basepath=os.path.abspath('.'), parentpath='', name='', dst_dir=''):
            fullpath = os.path.join(basepath, parentpath, name)
            for fname in os.listdir(fullpath):
                ffile = os.path.join(fullpath, fname)
                if os.path.isdir(ffile) and fname != build_dir and (not fname.startswith('.'))\
                        and fname != '__pycache__':
                    self.copy_file_recusive(basepath, os.path.join(parentpath, name), fname, dst_dir)
                elif os.path.isfile(ffile):
                    if os.path.splitext(fname)[1] in ('.py', '.pyx'):
                         if MyBuildExt.manifest is None:
                             # all py pyx compiled into so, except __init__.py
                             if fname == '__init__.py':
                                 dstdir = os.path.join(basepath, dst_dir, parentpath, name)
                                 if not os.path.isdir(dstdir): os.makedirs(dstdir)
                                 shutil.copyfile(ffile, os.path.join(dstdir, fname))
                         else:
                             pack_file_path = os.path.join('./acuitynet', parentpath, name, fname)
                             if pack_file_path in MyBuildExt.manifest.keys():
                                    if MyBuildExt.build_package_type in ['acuity-src-cpu', 'acuity-src-gpu'] \
                                        and MyBuildExt.manifest[pack_file_path].forcetype == 'bin':
                                        #DO NOT copy files that compiled into so
                                        pass
                                    elif MyBuildExt.build_package_type not in ['acuity-src-cpu', 'acuity-src-gpu'] \
                                            and fname != '__init__.py'\
                                            and MyBuildExt.manifest[pack_file_path].forcetype != 'src':
                                        #whl bin package, DO NOT copy files that compiled into so,
                                        #__init__.py haven't compiled into so, need copy in the following 'else' path
                                        pass
                                    else:
                                        dstdir = os.path.join(basepath, dst_dir, parentpath, name)
                                        if not os.path.isdir(dstdir): os.makedirs(dstdir)
                                        shutil.copyfile(ffile, os.path.join(dstdir, fname))
                             else:
                                 pass

    def getpy(basepath=os.path.abspath('.'), parentpath='', name='', excepts=(),
              copyOther=False, delC=False, manifest=None, package_type=''):
        """
        get all py file path
        param basepath: root path
        param parentpath: parent path
        param name: file name or folder name
        param excepts: files need to be skipped
        param copy: does need copy other relative files
        return: py file iterator
        """
        fullpath = os.path.join(basepath, parentpath, name)
        for fname in os.listdir(fullpath):
            ffile = os.path.join(fullpath, fname)
            if os.path.isdir(ffile) and fname != build_dir and (not fname.startswith('.')) and fname != '__pycache__':
                for f in getpy(basepath, os.path.join(parentpath, name), fname, excepts, copyOther, delC,
                               manifest, package_type):
                    yield f
            elif os.path.isfile(ffile):
                ext = os.path.splitext(fname)[1]
                if ext == ".c":
                    if delC and os.stat(ffile).st_mtime > pack_start_time:
                        os.remove(ffile)
                elif ffile not in excepts and os.path.splitext(fname)[1] not in ('.pyc', '.pyx'):
                    if os.path.splitext(fname)[1] in ('.py', '.pyx') and not fname.startswith('__'):
                         if manifest is None:
                             # py pyx files that compiled into so
                             yield os.path.join(parentpath, name, fname)
                         else:
                             pack_file_path = os.path.join('./acuitynet', parentpath, name, fname)
                             if pack_file_path in manifest.keys():
                                 if package_type == 'src' and manifest[pack_file_path].forcetype == 'bin'\
                                    or (package_type != 'src' and manifest[pack_file_path].forcetype != 'src'):
                                     # py pyx files that compiled into so
                                     yield os.path.join(parentpath, name, fname)
                                 else:
                                     if copyOther:
                                         dstdir = os.path.join(basepath, build_dir, parentpath, name)
                                         if not os.path.isdir(dstdir): os.makedirs(dstdir)
                                         shutil.copyfile(ffile, os.path.join(dstdir, fname))
                             else:
                                 pass
                    else:
                        if copyOther:
                            if manifest is None:
                                #CAUTION __pycache__ maybe copied
                                dstdir = os.path.join(basepath, build_dir, parentpath, name)
                                if not os.path.isdir(dstdir): os.makedirs(dstdir)
                                shutil.copyfile(ffile, os.path.join(dstdir, fname))
                            else:
                                if os.path.join('./acuitynet', parentpath, name, fname) in manifest.keys():
                                    #CAUTION __pycache__ maybe copied

                                    #for whl bin
                                    dstdir1 = os.path.join(basepath,
                                                          'build/lib.linux-x86_64-{}.{}'.
                                                          format(sys.version_info.major, sys.version_info.minor),
                                                          parentpath, name)
                                    if not os.path.isdir(dstdir1): os.makedirs(dstdir1)
                                    shutil.copyfile(ffile, os.path.join(dstdir1, fname))

                                    #for whl src
                                    dstdir2 = os.path.join(basepath,
                                                          'build/lib',
                                                          parentpath, name)
                                    if not os.path.isdir(dstdir2): os.makedirs(dstdir2)
                                    shutil.copyfile(ffile, os.path.join(dstdir2, fname))
            else:
                pass

    print('Pack acuity {} package begin ...'.format(build_package_type))
    manifest_parse_results = None
    if verify_manifest == 'true':
        manifest_parse_results = OrderedDict()
        yaml_walker_get_pack_files(pack_edition,'', manifest_yaml_obj, manifest_parse_results, '.', False)

    #get pys that need to be compiled into so
    # package_type == src, means whl_src package
    # package_type == bin, means whl_bin package and binary package
    src_module_list = list(getpy(basepath=rootdir, parentpath=parentpath, excepts=(setupfile),
                             manifest=manifest_parse_results, package_type='src'))
    bin_module_list = list(getpy(basepath=rootdir, parentpath=parentpath, excepts=(setupfile),
                             manifest=manifest_parse_results, package_type='bin'))
    try:

        MyBuildExt.set_manifest(manifest_parse_results)

        acuity_src_cpu_test_setup_configs = [
            {
                'name': 'acuity-src-cpu_test',
                'version': version,
                'author': 'Verisilicon',
                'author_email': 'support@verisilicon.com',
                'description': 'acuity-src-cpu test wheel package',
                'long_description': long_description,
                'long_description_content_type': 'text/markdown',
                'packages': setuptools.find_packages(),
                'install_requires': REQUIRED_PACKAGES_CPU_VERSION,
                'package_data': CODE_TEMPLATE,
                'data_files': ACUITY_COMMAND_LINE_TOOLS,
                'classifiers': [
                    "Programming Language :: Python :: 3",
                    "Operating System :: OS Indenpendent",
                ],
            },
        ]
        acuity_src_gpu_test_setup_configs = [
            {
                'name': 'acuity-src-gpu_test',
                'version': version,
                'author': 'Verisilicon',
                'author_email': 'support@verisilicon.com',
                'description': 'acuity-src-gpu test wheel package',
                'long_description': long_description,
                'long_description_content_type': 'text/markdown',
                'packages': setuptools.find_packages(),
                'install_requires': REQUIRED_PACKAGES_GPU_VERSION,
                'package_data': CODE_TEMPLATE,
                'data_files': ACUITY_COMMAND_LINE_TOOLS,
                'classifiers': [
                    "Programming Language :: Python :: 3",
                    "Operating System :: OS Indenpendent",
                ],
            },
        ]

        acuity_src_cpu_pure_setup_configs = [
            {
                'name': 'acuity-src-cpu',
                'version': version,
                'author': 'Verisilicon',
                'author_email': 'support@verisilicon.com',
                'description': 'acuity-src-cpu wheel package',
                'long_description': long_description,
                'long_description_content_type': 'text/markdown',
                'packages': setuptools.find_packages(),
                'install_requires': REQUIRED_PACKAGES_CPU_VERSION,
                'package_data': CODE_TEMPLATE,
                'data_files': ACUITY_COMMAND_LINE_TOOLS,
                'classifiers': [
                    "Programming Language :: Python :: 3",
                    "Operating System :: OS Indenpendent",
                ],
            },
        ]
        acuity_src_gpu_pure_setup_configs = [
            {
                'name': 'acuity-src-gpu',
                'version': version,
                'author': 'Verisilicon',
                'author_email': 'support@verisilicon.com',
                'description': 'acuity-src-gpu wheel package',
                'long_description': long_description,
                'long_description_content_type': 'text/markdown',
                'packages': setuptools.find_packages(),
                'install_requires': REQUIRED_PACKAGES_GPU_VERSION,
                'package_data': CODE_TEMPLATE,
                'data_files': ACUITY_COMMAND_LINE_TOOLS,
                'classifiers': [
                    "Programming Language :: Python :: 3",
                    "Operating System :: OS Indenpendent",
                ],
            },
        ]

        acuity_src_cpu_setup_configs = [
            {
                'name': 'acuity-src-cpu-ex',
                'version': version,
                'author': 'Verisilicon',
                'author_email': 'support@verisilicon.com',
                'description': 'acuity-src-cpu wheel package',
                'long_description': long_description,
                'long_description_content_type': 'text/markdown',
                'packages': [],
                'install_requires': REQUIRED_PACKAGES_CPU_VERSION,
                'package_data': CODE_TEMPLATE,
                'data_files': ACUITY_COMMAND_LINE_TOOLS,
                'ext_modules': cythonize(src_module_list,compiler_directives={'language_level': 3}),
                'cmdclass': dict(build_ext=MyBuildExt),
            },
        ]
        acuity_src_gpu_setup_configs = [
            {
                'name': 'acuity-src-gpu-ex',
                'version': version,
                'author': 'Verisilicon',
                'author_email': 'support@verisilicon.com',
                'description': 'acuity-src-gpu wheel package',
                'long_description': long_description,
                'long_description_content_type': 'text/markdown',
                'packages': setuptools.find_packages(),
                'install_requires': REQUIRED_PACKAGES_GPU_VERSION,
                'package_data': CODE_TEMPLATE,
                'data_files': ACUITY_COMMAND_LINE_TOOLS,
                'ext_modules': cythonize(src_module_list,compiler_directives={'language_level': 3}),
                'cmdclass': dict(build_ext=MyBuildExt),
            },
        ]
        acuity_bin_cpu_setup_configs = [
            {
                'name': 'acuity-bin-cpu',
                'version': version,
                'author': 'Verisilicon',
                'author_email': 'support@verisilicon.com',
                'description': 'acuity-bin-cpu wheel binary package',
                'long_description': long_description,
                'long_description_content_type': 'text/markdown',
                'packages': [],
                'install_requires': REQUIRED_PACKAGES_CPU_VERSION,
                'package_data': CODE_TEMPLATE,
                'data_files': ACUITY_COMMAND_LINE_TOOLS,
                'ext_modules': cythonize(bin_module_list,compiler_directives={'language_level': 3}),
                'cmdclass': dict(build_ext=MyBuildExt),
            },
        ]
        acuity_bin_gpu_setup_configs = [
            {
                'name': 'acuity-bin-gpu',
                'version': version,
                'author': 'Verisilicon',
                'author_email': 'support@verisilicon.com',
                'description': 'acuity-bin-gpu wheel binary package',
                'long_description': long_description,
                'long_description_content_type': 'text/markdown',
                'packages': [],
                'install_requires': REQUIRED_PACKAGES_GPU_VERSION,
                'package_data': CODE_TEMPLATE,
                'data_files': ACUITY_COMMAND_LINE_TOOLS,
                'ext_modules': cythonize(bin_module_list, compiler_directives={'language_level': 3}),
                'cmdclass': dict(build_ext=MyBuildExt),
            },
        ]
        setup_configs = []
        if build_package_type == 'build_test':
            setup_configs = acuity_src_cpu_test_setup_configs
        elif build_package_type == 'build_whl_src':
            setup_configs = acuity_src_cpu_pure_setup_configs + acuity_src_gpu_pure_setup_configs
        elif build_package_type == 'build_whl_bin':
            setup_configs = acuity_bin_cpu_setup_configs + acuity_bin_gpu_setup_configs
        elif build_package_type in ['build_binary_bin', 'build_dev', 'build_rel']:
            pass
        else:
            print("Error: Unknow build package type {}.".format(build_package_type))

        print('Copy related files begin')
        module_list = list(getpy(basepath=rootdir, parentpath='acuitylib', excepts=(setupfile), copyOther=True,
                                 manifest=manifest_parse_results))
        print('Copy related files done.')

        for setup_config in setup_configs:
            name = setup_config['name']
            print("Packing '{}' ...".format(name))
            MyBuildExt.set_build_package_type(name)
            p = Process(target=setup, kwargs=setup_config)
            p.start()
            p.join()
            print("Pack '{}' wheel package done".format(name))
    except Exception as e:
        print("Pack acuity wheel package error! -> {}".format(e))
        module_list = list(getpy(basepath=rootdir, parentpath=parentpath, excepts=(setupfile), delC=True,
                                 manifest=manifest_parse_results))
        print('Cleaned temporary c files.')

    copy_command_line_tools(ACUITY_COMMAND_LINE_TOOLS)

    module_list = list(getpy(basepath=rootdir, parentpath=parentpath, excepts=(setupfile), delC=True,
                             manifest=manifest_parse_results))
    if os.path.exists(build_tmp_dir): shutil.rmtree(build_tmp_dir)
    print('Clean temporary c files done.')

    print('Pack acuity {} package done.'.format(build_package_type))

if __name__ == '__main__':
    main()
