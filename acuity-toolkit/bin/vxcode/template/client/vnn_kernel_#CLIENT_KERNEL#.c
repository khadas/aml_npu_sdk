#include <VX/vx_khr_cnn.h>
#include <VX/vx_helper.h>
#include <VX/vx.h>
#include <VX/vx_ext_program.h>

#include "vsi_nn_pub.h"
#include "client/vsi_nn_vxkernel.h"
#include "vnn_#NETWORK_LOWER#_client_tab.h"

#define _VX_KERNEL_ID           (VX_CLIENT_ID_#CLIENT_KERNEL_OP#)
#define _VX_KERNEL_VAR_CPU      (vx_client_kernel_#CLIENT_KERNEL_OP#_CPU)
#define _VX_KERNEL_VAR_VX       (vx_client_kernel_#CLIENT_KERNEL_OP#_VX)
#define _VX_KERNEL_NAME         ("com.vivantecorp.extension.#CLIENT_KERNEL_OP_STD#VXC")
#define _VX_KERNEL_FUNC_KERNEL  (vx#CLIENT_KERNEL_OP_STD#Kernel)

static vsi_status VX_CALLBACK vx#CLIENT_KERNEL_OP_STD#Kernel
    (
    vx_node node,
    const vx_reference* paramObj,
    uint32_t paramNum
    )
{
    /* TODO: Add CPU kernel implement */
    return VSI_SUCCESS;
} /* _VX_KERNEL_FUNC_KERNEL() */

static vx_param_description_t s_params[] =
    {
#CLIENT_KERNEL_PARAMETER_DECLARATIONS#
    };

static vx_status VX_CALLBACK vx#CLIENT_KERNEL_OP_STD#Initializer
    (
    vx_node nodObj,
    const vx_reference *paramObj,
    vx_uint32 paraNum
    )
{
    vx_status status = VX_SUCCESS;
    /*TODO: Add initial code for VX program*/

    return status;
}


#ifdef __cplusplus
extern "C" {
#endif
vx_kernel_description_t _VX_KERNEL_VAR_CPU =
{
    _VX_KERNEL_ID,
    _VX_KERNEL_NAME,
    _VX_KERNEL_FUNC_KERNEL,
    s_params,
    _cnt_of_array( s_params ),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vsi_nn_KernelInitializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t _VX_KERNEL_VAR_VX =
{
    _VX_KERNEL_ID,
    _VX_KERNEL_NAME,
    NULL,
    s_params,
    _cnt_of_array( s_params ),
    vsi_nn_KernelValidator,
    NULL,
    NULL,
    vx#CLIENT_KERNEL_OP_STD#Initializer,
    vsi_nn_KernelDeinitializer
};

vx_kernel_description_t * vx_kernel_#CLIENT_KERNEL_OP#_list[] =
{
    &_VX_KERNEL_VAR_CPU,
    &_VX_KERNEL_VAR_VX,
    NULL
};
#ifdef __cplusplus
}
#endif
