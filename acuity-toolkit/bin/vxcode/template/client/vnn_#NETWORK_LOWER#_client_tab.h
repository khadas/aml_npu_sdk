#ifndef _VNN_#NETWORK_UPPER#_CLIENT_TAB_H
#define _VNN_#NETWORK_UPPER#_CLIENT_TAB_H

#include <VX/vx.h>
#include "vsi_nn_pub.h"
#CLIENT_OP_HEADERS#

enum
{
    _CLIENT_OP_START = VSI_NN_OP_CLIENT,
    #define DEF_OP( name )     VSI_NN_OP_CLIENT_##name,
#DEF_OPS_TAB_1#
    #undef DEF_OP
    VSI_NN_OP_CLIENT_NUM
};

/* Assigned from Khronos, */
#define VX_LIBRARY_DH (0x3)
enum
{
    _VX_CLIENT_ID_START = VX_KERNEL_BASE( VX_ID_DEFAULT, VX_LIBRARY_DH ) - 1,
    #define DEF_OP( name )     VX_CLIENT_ID_##name,
#DEF_OPS_TAB_1#
    #undef DEF_OP
};

vsi_bool vsi_nn_Register#NETWORK_STD#ClientOps
    (
    vsi_nn_graph_t * graph
    );

vsi_bool vsi_nn_Unregister#NETWORK_STD#ClientOps
    (
    vsi_nn_graph_t * graph
    );

#endif
