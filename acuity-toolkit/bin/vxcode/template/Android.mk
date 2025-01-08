# Build Vivante chipinfo for android.
#
LOCAL_PATH:= $(call my-dir)
include $(CLEAR_VARS)

include $(AQROOT)/Android.mk.def

ifeq ($(PLATFORM_VENDOR),1)
LOCAL_VENDOR_MODULE  := true
endif


# head file
LOCAL_SRC_FILES :=\
    main.c \
    vnn_post_process.c \
    vnn_pre_process.c \
    vnn_#NETWORK_NAME_LOWER#.c \

LOCAL_C_INCLUDES += \
        external/libjpeg-turbo \
    $(AQROOT)/sdk/inc/CL \
    $(AQROOT)/sdk/inc/VX \
    $(LOCAL_PATH) \
    $(OVXLIB_DIR)/include \
    $(OVXLIB_DIR)/include/ops \
    $(OVXLIB_DIR)/include/utils \
    $(OVXLIB_DIR)/include/infernce \
    $(OVXLIB_DIR)/include/platform \
    $(OVXLIB_DIR)/include/client \
    $(OVXLIB_DIR)/include/libnnext\
    $(AQROOT)/sdk/inc


LOCAL_CFLAGS :=  \
        -Werror \
        -D'OVXLIB_API=__attribute__((visibility("default")))' \
        -Wno-sign-compare \
        -Wno-implicit-function-declaration \
        -Wno-sometimes-uninitialized \
        -Wno-unused-parameter \
        -Wno-enum-conversion \
        -Wno-missing-field-initializers \
        -Wno-tautological-compare \
        -Wno-format \


LOCAL_SHARED_LIBRARIES += \
        libovxlib \
        libOpenVX \
        libVSC\
        libGAL\

LOCAL_STATIC_LIBRARIES := libjpeg

LOCAL_MODULE:=#NETWORK_NAME_LOWER#
LOCAL_MODULE_TAGS := optional

include $(BUILD_EXECUTABLE)

