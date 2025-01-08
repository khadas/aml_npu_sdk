#!/bin/bash
#BUILD_PACKAGE_TYPE in  build_test, build_dev, build_rel, build_whl_bin, build_whl_src, build_binary_bin

BUILD_PACKAGE_TYPE="build_whl_bin"
BUILD_PACKAGE_EDITION="pro"
VERIFY_MANIFEST="true"

if [ $# -eq 1 ]; then
    BUILD_PACKAGE_TYPE=$1
elif [ $# -eq 2 ]; then
    BUILD_PACKAGE_TYPE=$1
    BUILD_PACKAGE_EDITION=$2
elif [ $# -eq 3 ]; then
    BUILD_PACKAGE_TYPE=$1
    BUILD_PACKAGE_EDITION=$2
    VERIFY_MANIFEST=$3
fi

if [ "$VERIFY_MANIFEST" = "true" ]; then
    # verify manifest
    python3 ../verify_manifest.py .. ../manifest.yml
fi

if [ $? -ne 0 ]; then
    exit -1
else
    python3 setup.py "$BUILD_PACKAGE_TYPE" "$BUILD_PACKAGE_EDITION" "$VERIFY_MANIFEST" sdist bdist_wheel
    if [ "$BUILD_PACKAGE_TYPE" = "build_binary_bin" ]; then
        pyinstaller acuity.spec
        find -name "*_pb2.py" -exec cp  {} ./dist/ \;
    fi
    exit 0
fi


