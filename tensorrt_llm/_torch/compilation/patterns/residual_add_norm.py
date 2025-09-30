from operator import getitem

import torch
from torch._inductor.pattern_matcher import (MULTIPLE, CallFunction, KeywordArg,
                                             Match, MultiOutputPattern,
                                             PatternMatcherPass, fwd_only,
                                             register_replacement)

aten = torch.ops.aten
from torch._higher_order_ops.auto_functionalize import auto_functionalized


def register_add_norm(custom_pass: PatternMatcherPass):
    residual = KeywordArg("residual")
    add_Tensor = CallFunction(aten.add.Tensor,
                              KeywordArg("input"),
                              residual,
                              _users=MULTIPLE)
    flashinfer_norm_default = CallFunction(
        torch.ops.trtllm.flashinfer_rmsnorm.default,
        add_Tensor,
        KeywordArg("norm_weight"),
        KeywordArg("eps"),
        _users=MULTIPLE)
    add_norm_pattern = MultiOutputPattern([flashinfer_norm_default, add_Tensor])

    def empty_pattern(
        input: torch.Tensor,
        residual: torch.Tensor,
        norm_weight: torch.nn.Parameter,
        eps: float,
    ):
        return

    def target_pattern(
        input: torch.Tensor,
        residual: torch.Tensor,
        norm_weight: torch.nn.Parameter,
        eps: float,
    ):
        at = auto_functionalized(
            torch.ops.trtllm.flashinfer_fused_add_rmsnorm.default,
            input=input,
            residual=residual,
            weight=norm_weight,
            eps=eps)
        return at[1], at[2]

    def extra_check(match: Match):
        # Check the original residual and hidden has no other users since we will inplace update them
        residual_node = match.ctx.pattern_to_node[add_Tensor]
        if not isinstance(residual_node, torch.fx.graph.Node):
            return False

        # torch uses dict here to guarantee the order of the uses
        if list(residual_node.args[0].users.keys()
                )[-1] != residual_node or list(
                    residual_node.args[1].users.keys())[-1] != residual_node:
            return False

        return True

    register_replacement(
        empty_pattern,
        target_pattern,
        [],
        fwd_only,
        custom_pass,
        search_fn_pattern=add_norm_pattern,
        extra_check=extra_check,
    )


#    %add_26 : [num_users=2] = call_function[target=operator.add](args = (%output_10, %add_25), kwargs = {})

#    %flashinfer_rmsnorm_default_3 : [num_users=1] = call_function[target=torch.ops.trtllm.flashinfer_rmsnorm.default](args = (%add_26, %l_self_mo    dules_layers_modules_1_modules_post_attention_layernorm_parameters_weight_, 1e-06), kwargs = {})

#    %static_quantize_e4m3_per_tensor_3 : [num_users=1] = call_function[target=torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor](args = (%fl    ashinfer_rmsnorm_default_3, %l_self_modules_layers_modules_1_modules_mlp_modules_gate_up_proj_parameters_input_scale_), kwargs = {})

#    %qinput_3 : [num_users=1] = call_function[target=operator.getitem](args = (%static_quantize_e4m3_per_tensor_3, 0), kwargs = {})

#    %t_6 : [num_users=1] = call_method[target=t](args = (%l_self_modules_layers_modules_1_modules_mlp_modules_gate_up_proj_parameters_weight_,),     kwargs = {})

#    %output_11 : [num_users=1] = call_function[target=torch.ops.trtllm.cublas_scaled_mm](args = (%qinput_3, %t_6)


def register_add_norm_quant_fp8(custom_pass: PatternMatcherPass):
    residual_out = CallFunction(aten.add.Tensor,
                                KeywordArg("input"),
                                KeywordArg("residual"),
                                _users=MULTIPLE)

    flashinfer_norm_default = CallFunction(
        torch.ops.trtllm.flashinfer_rmsnorm.default,
        residual_out,
        KeywordArg("norm_weight"),
        KeywordArg("eps"),
        _users=MULTIPLE)

    static_quantize = CallFunction(
        torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor.default,
        flashinfer_norm_default,
        KeywordArg("scale"),
        _users=MULTIPLE)

    quant_out = CallFunction(getitem, static_quantize, 0, _users=1)
    scale = CallFunction(getitem, static_quantize, 1, _users=1)

    add_norm_quant_pattern = MultiOutputPattern(
        [quant_out, residual_out, scale])

    def empty_pattern(
        input: torch.Tensor,
        residual: torch.Tensor,
        norm_weight: torch.nn.Parameter,
        eps: float,
        scale: torch.Tensor,
    ):
        return

    def target_pattern(
        input: torch.Tensor,
        residual: torch.Tensor,
        norm_weight: torch.nn.Parameter,
        eps: float,
        scale: torch.Tensor,
    ):
        at = torch.ops.trtllm.rms_norm_quant_fp8.default(
            input=input,
            residual=residual,
            norm_weight=norm_weight,
            eps=eps,
            scale=scale)
        return at[0], at[1], scale

    def extra_check(match: Match) -> bool:
        return True

    register_replacement(
        empty_pattern,
        target_pattern,
        [],
        fwd_only,
        custom_pass,
        search_fn_pattern=add_norm_quant_pattern,
        extra_check=extra_check,
    )
