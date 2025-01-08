import sys
import platform
import importlib
using_distro = False
try:
    import distro
    using_distro = True
except ImportError:
    pass
from argparse import ArgumentParser

def main():
    options = ArgumentParser(description='Acuity environment check tool.')

    options.add_argument('--distribution',
                        action='store_true',
                        help='if it is a distribution environment')
    args = options.parse_args()

    check_status = check_env(args.distribution)
    if check_status is True:
        sys.exit(0)
    else:
        sys.exit(-1)

def check_env(distribution_env = False):
    env_check_status = True

    # Check OS
    if using_distro:
        distribution = distro.linux_distribution()
    else:
        distribution = platform.linux_distribution()
    linux_os = distribution[0]
    linux_os_ver = distribution[1]
    python_ver = platform.python_version()
    require_python_ver_major = '0'
    require_python_ver_minor = '0'
    require_pyinstaller_version = ''
    if linux_os.lower() == 'ubuntu':
        if linux_os_ver == '16.04':
            require_python_ver_major = 3
            require_python_ver_minor = 5
            require_pyinstaller_version = '3.4'
            print(
                "Env Info: Linux distribution is {} {}, please make sure python {}.{} installed." \
                .format(linux_os, linux_os_ver, require_python_ver_major, require_python_ver_minor), flush=True)
        elif linux_os_ver == '18.04':
            require_python_ver_major = 3
            require_python_ver_minor = 6
            require_pyinstaller_version = '3.4'
            print(
                "Env Info: Linux distribution is {} {}, please make sure python {}.{}  installed properly." \
                .format(linux_os, linux_os_ver, require_python_ver_major, require_python_ver_minor), flush=True)
        elif linux_os_ver == '20.04':
            require_python_ver_major = 3
            require_python_ver_minor = 8
            require_pyinstaller_version = '4.5.1'
            print(
                "Env Warning: Linux distribution is {} {}, please make sure python {}.{}  installed properly." \
                .format(linux_os, linux_os_ver, require_python_ver_major, require_python_ver_minor), flush=True)
        else:
            env_check_status = False
            print(
                "Env Error: Linux distribution is {} {}, python is {}, not supported." \
                    .format(linux_os, linux_os_ver, python_ver), flush=True)

        #Check mainche type, OS type
        platform_machine = platform.machine()
        platform_architecture = platform.architecture()
        if not platform_machine.endswith('64'):
            env_check_status = False
            print('Env Error: Require machine type 64bit, but {} installed'.format(platform_architecture), flush=True)
        else:
            print('Env Pass: Require machine type 64bit installed', flush=True)
        if platform_architecture[0] != '64bit':
            env_check_status = False
            print('Env Error: Require OS type 64bit, but {} installed'.format(platform_architecture[0]), flush=True)
        else:
            print('Env Pass: Require OS type 64bit installed', flush=True)

        # Check Processor avx instruction set.
        avx_support = True
        with open('/proc/cpuinfo', 'r') as f:
            processor = ''
            for line in f.readlines():
                line = line.strip()
                if len(line) == 0:
                    continue
                key,val = line.split(':')
                key = key.strip()
                val = val.strip()
                if 'processor' == key:
                    processor = val
                if 'flags' == key:
                    sets = val.split(' ')
                    if 'avx' not in sets:
                        avx_support = False
                        print('Env Error: Processor {} not support instruction set avx'.format(processor), flush=True)
        if avx_support:
            print('Env Pass: All Processors support instruction set avx'.format(processor), flush=True)
        else:
            env_check_status = False
            print('Env Error: Not All Processors support instruction set avx'.format(processor), flush=True)
            print('Env Error: Not All Processors support instruction set avx'.format(processor), flush=True)

    else:
        print("Env Error: Linux distribution is {} {}, Not supported.".format(linux_os, linux_os_ver),
              flush=True)
        env_check_status = False

    # check python version
    if python_ver >= "{}.{}.{}".format(require_python_ver_major, require_python_ver_minor, 0) \
            and python_ver < "{}.{}".format(require_python_ver_major, require_python_ver_minor + 1):
        print("Env Pass: Python {} installed, require {}.{}.x"
              .format(python_ver, require_python_ver_major, require_python_ver_minor), flush=True)
    else:
        print("Env Error: Python {} installed, require {}.{}.x"
              .format(python_ver, require_python_ver_major, require_python_ver_minor), flush=True)
        env_check_status = False

    # check python library version
    require_libs = ['PyInstaller=={}'.format(require_pyinstaller_version), 'Cython==0.29','setuptools>=42.0.2',
                    'tensorflow==2.3.0',
                    'networkx>=1.11', 'lmdb==0.93', 'onnx==1.8.1',
                    'flatbuffers==1.10', 'dill==0.2.8.2', 'ruamel.yaml==0.15.81',
                    'ply==3.11', 'torch==1.5.1']
    non_distribution_env_ignore_libs = ['PyInstaller', 'Cython', 'setuptools']

    for lib in require_libs:
        require_lib_name = ''
        require_lib_compare_sign = ''
        require_lib_ver = ''
        if '==' in lib:
            require_lib_compare_sign = '=='
            lib_name_list = lib.split('==')
        elif '>=' in lib:
            require_lib_compare_sign = '>='
            lib_name_list = lib.split('>=')
        elif '<=' in lib:
            require_lib_compare_sign = '<='
            lib_name_list = lib.split('<=')
        elif '>' in lib:
            require_lib_compare_sign = '>'
            lib_name_list = lib.split('>')
        elif '<' in lib:
            require_lib_compare_sign = '<'
            lib_name_list = lib.split('<')
        else:
            raise ValueError()
        if len(lib_name_list) == 1:
            require_lib_name = lib_name_list[0]
        elif len(lib_name_list) == 2:
            require_lib_name = lib_name_list[0]
            require_lib_ver = lib_name_list[1]
        else:
            print("Env Error: Parse require libs failed, unexpected segment.", flush=True)
            env_check_status = False

        if require_lib_name != '':
            if not distribution_env and require_lib_name in non_distribution_env_ignore_libs:
                continue  #ignore libs in ignore_libs for non-distribuition-env

            python_lib = None

            try:
                python_lib = importlib.import_module(require_lib_name)
            except:
                print("Env Error: Require python lib {} import testing failed, not installed."
                      .format(require_lib_name), flush=True)
                env_check_status = False

            machine_python_lib_ver = ''
            try:
                if require_lib_name == 'image':
                    machine_python_lib_ver = python_lib.VERSION
                    machine_python_lib_ver = '.'.join(str(v) for v in machine_python_lib_ver)
                elif require_lib_name in ['flatbuffers']:
                    # FIX ME, if it is installed we think it's OK
                    machine_python_lib_ver = require_lib_ver
                else:
                    machine_python_lib_ver = python_lib.__version__
            except:
                print("Env Error: Get Require python lib {} Version failed, please check build script."
                      .format(require_lib_name), flush=True)
                env_check_status = False

            if require_lib_ver != '' and \
                    eval("'{}' {} '{}'".format(machine_python_lib_ver, require_lib_compare_sign, require_lib_ver)):
                print("Env Pass: Require python lib {} version {} {}, {} installed."
                      .format(require_lib_name, require_lib_compare_sign, require_lib_ver, machine_python_lib_ver),
                      flush=True)
            else:
                print("Env Error: Require python lib {} version {} {}, {} installed."
                      .format(require_lib_name, require_lib_compare_sign, require_lib_ver, machine_python_lib_ver),
                      flush=True)
                env_check_status = False
        else:
            print("Env Error: Parse require libs failed, library name is null.", flush=True)
            env_check_status = False

    if env_check_status is False:
        print("Env Error: Env check FAILED!!!", flush=True)
    else:
        print("Env Pass: Env check SUCCESS!!!", flush=True)

    return env_check_status

if __name__ == '__main__':
    main()
