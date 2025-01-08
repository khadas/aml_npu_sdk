#include "vsi_nn_pub.h"
#include "vnn_#NETWORK_LOWER#_client_tab.h"


#define DEF_OP( name )    extern vsi_nn_op_proc_t vsi_nn_op_CLIENT_##name;
#DEF_OPS_NO_SPACE#
#undef DEF_OP

static vsi_nn_op_proc_t * s_client_ops[] =
    {
    #define DEF_OP( name )     &vsi_nn_op_CLIENT_##name,
#DEF_OPS_TAB_1#
    #undef DEF_OP
    };

static vsi_nn_op_t s_client_ops_id[] =
    {
    #define DEF_OP( name )     VSI_NN_OP_CLIENT_##name,
#DEF_OPS_TAB_1#
    #undef DEF_OP
    };

vsi_bool vsi_nn_Register#NETWORK_STD#ClientOps
    (
    vsi_nn_graph_t * graph
    )
{
    uint32_t i;
    vsi_bool ret;
    ret = TRUE;
    for( i = 0; i < _cnt_of_array( s_client_ops ); i++ )
    {
        ret = vsi_nn_OpRegisterClient( s_client_ops_id[i],
                s_client_ops[i] );
        if( FALSE == ret )
        {
            break;
        }
    }
    return ret;
} /* vsi_nn_Register#NETWORK_STD#ClientOps */

vsi_bool vsi_nn_Unregister#NETWORK_STD#ClientOps
    (
    vsi_nn_graph_t * graph
    )
{
    uint32_t i;
    vsi_bool ret;

    ret = TRUE;
    for( i = 0; i < _cnt_of_array( s_client_ops ); i++ )
    {
        vsi_nn_OpRemoveClient( s_client_ops_id[i]);
    }
    return ret;
} /* vsi_nn_Unregister#NETWORK_STD#ClientOps() */

