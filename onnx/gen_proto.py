#!/usr/bin/env python

# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import re
import glob
import subprocess
from textwrap import dedent
from typing import Iterable

autogen_header = """\
//
// WARNING: This file is automatically generated!  Please edit onnx.in.proto.
//


"""

LITE_OPTION = '''

// For using protobuf-lite
option optimize_for = LITE_RUNTIME;

'''

DEFAULT_PACKAGE_NAME = "onnx"

IF_ONNX_ML_REGEX = re.compile(r'\s*//\s*#if\s+ONNX-ML\s*$')
ENDIF_ONNX_ML_REGEX = re.compile(r'\s*//\s*#endif\s*$')
ELSE_ONNX_ML_REGEX = re.compile(r'\s*//\s*#else\s*$')


def process_ifs(lines: Iterable[str], onnx_ml: bool) -> Iterable[str]:
    in_if = 0
    for line in lines:
        if IF_ONNX_ML_REGEX.match(line):
            assert 0 == in_if
            in_if = 1
        elif ELSE_ONNX_ML_REGEX.match(line):
            assert 1 == in_if
            in_if = 2
        elif ENDIF_ONNX_ML_REGEX.match(line):
            assert (1 == in_if or 2 == in_if)
            in_if = 0
        else:
            if 0 == in_if:
                yield line
            elif (1 == in_if and onnx_ml):
                yield line
            elif (2 == in_if and not onnx_ml):
                yield line


IMPORT_REGEX = re.compile(r'(\s*)import\s*"([^"]*)\.proto";\s*$')
PACKAGE_NAME_REGEX = re.compile(r'\{PACKAGE_NAME\}')
ML_REGEX = re.compile(r'(.*)\-ml')


def process_package_name(lines: Iterable[str], package_name: str) -> Iterable[str]:
    need_rename = (package_name != DEFAULT_PACKAGE_NAME)
    for line in lines:
        m = IMPORT_REGEX.match(line) if need_rename else None
        if m:
            include_name = m.group(2)
            ml = ML_REGEX.match(include_name)
            if ml:
                include_name = f"{ml.group(1)}_{package_name}-ml"
            else:
                include_name = f"{include_name}_{package_name}"
            yield m.group(1) + f'import "{include_name}.proto";'
        else:
            yield PACKAGE_NAME_REGEX.sub(package_name, line)


PROTO_SYNTAX_REGEX = re.compile(r'(\s*)syntax\s*=\s*"proto2"\s*;\s*$')
OPTIONAL_REGEX = re.compile(r'(\s*)optional\s(.*)$')


def convert_to_proto3(lines: Iterable[str]) -> Iterable[str]:
    for line in lines:
        # Set the syntax specifier
        m = PROTO_SYNTAX_REGEX.match(line)
        if m:
            yield m.group(1) + 'syntax = "proto3";'
            continue

        # Remove optional keywords
        m = OPTIONAL_REGEX.match(line)
        if m:
            yield m.group(1) + m.group(2)
            continue

        # Rewrite import
        m = IMPORT_REGEX.match(line)
        if m:
            yield m.group(1) + f'import "{m.group(2)}.proto3";'
            continue

        yield line


def gen_proto3_code(protoc_path: str, proto3_path: str, include_path: str, cpp_out: str, python_out: str) -> None:
    print(f"Generate pb3 code using {protoc_path}")
    build_args = [protoc_path, proto3_path, '-I', include_path]
    build_args.extend(['--cpp_out', cpp_out, '--python_out', python_out])
    subprocess.check_call(build_args)


def translate(source: str, proto: int, onnx_ml: bool, package_name: str) -> str:
    lines: Iterable[str] = source.splitlines()
    lines = process_ifs(lines, onnx_ml=onnx_ml)
    lines = process_package_name(lines, package_name=package_name)
    if proto == 3:
        lines = convert_to_proto3(lines)
    else:
        assert proto == 2
    return "\n".join(lines)  # TODO: not Windows friendly


def qualify(f: str, pardir: str = os.path.realpath(os.path.dirname(__file__))) -> str:
    return os.path.join(pardir, f)


def convert(stem: str, package_name: str, output: str, do_onnx_ml: bool = False, lite: bool = False, protoc_path: str = '') -> None:
    proto_in = qualify(f"{stem}.in.proto")
    need_rename = (package_name != DEFAULT_PACKAGE_NAME)
    # Having a separate variable for import_ml ensures that the import statements for the generated
    # proto files can be set separately from the ONNX_ML environment variable setting.
    import_ml = do_onnx_ml
    # We do not want to generate the onnx-data-ml.proto files for onnx-data.in.proto,
    # as there is no change between onnx-data.proto and the ML version.
    if 'onnx-data' in proto_in:
        do_onnx_ml = False
    if do_onnx_ml:
        proto_base = f"{stem}_{package_name}-ml" if need_rename else f"{stem}-ml"
    else:
        proto_base = f"{stem}_{package_name}" if need_rename else f"{stem}"
    proto = qualify(f"{proto_base}.proto", pardir=output)
    proto3 = qualify(f"{proto_base}.proto3", pardir=output)

    print(f"Processing {proto_in}")
    with open(proto_in) as fin:
        source = fin.read()
        print(f"Writing {proto}")
        with open(proto, 'w', newline='') as fout:
            fout.write(autogen_header)
            fout.write(translate(source, proto=2, onnx_ml=import_ml, package_name=package_name))
            if lite:
                fout.write(LITE_OPTION)
        print(f"Writing {proto3}")
        with open(proto3, 'w', newline='') as fout:
            fout.write(autogen_header)
            fout.write(translate(source, proto=3, onnx_ml=import_ml, package_name=package_name))
            if lite:
                fout.write(LITE_OPTION)
        if protoc_path:
            porto3_dir = os.path.dirname(proto3)
            base_dir = os.path.dirname(porto3_dir)
            gen_proto3_code(protoc_path, proto3, base_dir, base_dir, base_dir)
            pb3_files = glob.glob(os.path.join(porto3_dir, '*.proto3.*'))
            for pb3_file in pb3_files:
                print(f"Removing {pb3_file}")
                os.remove(pb3_file)

        if need_rename:
            if do_onnx_ml:
                proto_header = qualify(f"{stem}-ml.pb.h", pardir=output)
            else:
                proto_header = qualify(f"{stem}.pb.h", pardir=output)
            print(f"Writing {proto_header}")
            with open(proto_header, 'w', newline='') as fout:
                fout.write("#pragma once\n")
                fout.write(f"#include \"{proto_base}.pb.h\"\n")

    # Generate py mapping
    # "-" is invalid in python module name, replaces '-' with '_'
    pb_py = qualify('{}_pb.py'.format(stem.replace('-', '_')), pardir=output)
    if need_rename:
        pb2_py = qualify('{}_pb2.py'.format(proto_base.replace('-', '_')), pardir=output)
    else:
        if do_onnx_ml:
            pb2_py = qualify('{}_ml_pb2.py'.format(stem.replace('-', '_')), pardir=output)
        else:
            pb2_py = qualify('{}_pb2.py'.format(stem.replace('-', '_')), pardir=output)

    print(f'generating {pb_py}')
    with open(pb_py, 'w') as f:
        f.write(str(dedent('''\
        # This file is generated by setup.py. DO NOT EDIT!


        from .{} import *  # noqa
        '''.format(os.path.splitext(os.path.basename(pb2_py))[0]))))


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Generates .proto file variations from .in.proto')
    parser.add_argument('-p', '--package', default='onnx',
                        help='package name in the generated proto files'
                        ' (default: %(default)s)')
    parser.add_argument('-m', '--ml', action='store_true', help='ML mode')
    parser.add_argument('-l', '--lite', action='store_true',
                        help='generate lite proto to use with protobuf-lite')
    parser.add_argument('-o', '--output',
                        default=os.path.realpath(os.path.dirname(__file__)),
                        help='output directory (default: %(default)s)')
    parser.add_argument('--protoc_path',
                        default='',
                        help='path to protoc for proto3 file validation')
    parser.add_argument('stems', nargs='*', default=['onnx', 'onnx-operators', 'onnx-data'],
                        help='list of .in.proto file stems '
                        '(default: %(default)s)')
    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    for stem in args.stems:
        convert(stem,
                package_name=args.package,
                output=args.output,
                do_onnx_ml=args.ml,
                lite=args.lite,
                protoc_path=args.protoc_path)


if __name__ == '__main__':
    main()
