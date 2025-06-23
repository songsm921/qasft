import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List, Union


class AsymmetricQuantizer(torch.autograd.Function):
    """
    FSDP-compatible implementation of AsymmetricQuantizer that matches fake_quantize_state_dict behavior.
    Ensures differentiability for quantization-aware training.
    """
    @staticmethod
    def forward(ctx, inputs, bits=8, scale=None, zero_point=None, 
                per_channel=False, channel_dim=0, 
                group_size=-1, group_dim=1,
                clipping_mode="none", clipping_ratio=1.0,
                beta=None, gamma=None):
        """
        Forward pass: simulate quantization by quantizing and dequantizing the inputs.
        Matches the behavior of fake_quantize_state_dict for consistent training and inference.
        """
        ctx.per_channel = per_channel
        ctx.channel_dim = channel_dim
        ctx.bits = bits
        ctx.group_size = group_size
        ctx.group_dim = group_dim
        ctx.clipping_mode = clipping_mode
        ctx.clipping_ratio = clipping_ratio
        
        # Keep original shape for later reshaping
        original_shape = inputs.shape
        
        # Handle group quantization
        if group_size > 0:
            # Reshape tensor to apply group quantization
            if group_dim == 1 and len(original_shape) == 2:
                # Common case for linear weights: [out_channels, in_channels]
                out_channels, in_features = original_shape
                # Calculate number of groups in the group_dim
                num_groups = in_features // group_size
                if in_features % group_size != 0:
                    # Pad to make group_size divide in_features evenly
                    pad_size = group_size - (in_features % group_size)
                    inputs = torch.nn.functional.pad(inputs, (0, pad_size))
                    in_features = in_features + pad_size
                    num_groups = in_features // group_size
                
                # Reshape to [out_channels, num_groups, group_size]
                inputs = inputs.reshape(out_channels, num_groups, group_size)
                
                # Find min/max per group
                min_val = inputs.min(dim=2, keepdim=True)[0]
                max_val = inputs.max(dim=2, keepdim=True)[0]
                
                # Apply clipping if needed
                if clipping_mode == "fixed_ratio":
                    # Adjust the quantization range by the fixed ratio
                    min_val = min_val * clipping_ratio
                    max_val = max_val * clipping_ratio
                elif clipping_mode == "learnable" and beta is not None and gamma is not None:
                    # Apply learnable clipping using sigmoid to ensure values in [0,1]
                    min_val = torch.sigmoid(beta).unsqueeze(-1) * min_val
                    max_val = torch.sigmoid(gamma).unsqueeze(-1) * max_val
                
                # Calculate scale and zero point
                qmin = 0
                qmax = 2**bits - 1
                scale = (max_val - min_val) / qmax
                
                # Add small epsilon to avoid division by zero (matching fake_quantize_state_dict)
                scale = scale + 1e-8
                
                zero_point = qmin - min_val / scale
                
                # Round zero point
                zero_point = torch.round(zero_point)
                
                # Clamp zero_point to valid range
                zero_point = torch.clamp(zero_point, qmin, qmax)
                
                # Apply quantization to each group
                inputs_flat = inputs.reshape(-1, group_size)
                #min_flat = min_val.reshape(-1, 1)
                #max_flat = max_val.reshape(-1, 1)
                scale_flat = scale.reshape(-1, 1)
                zero_point_flat = zero_point.reshape(-1, 1)
                
                # Quantize and dequantize
                inputs_q = torch.round(inputs_flat / scale_flat + zero_point_flat)
                inputs_q = torch.clamp(inputs_q, qmin, qmax)
                inputs_dq = (inputs_q - zero_point_flat) * scale_flat
                
                # Reshape back to original shape
                outputs = inputs_dq.reshape(out_channels, num_groups, group_size)
                outputs = outputs.reshape(out_channels, in_features)
                
                # Remove padding if added
                if in_features != original_shape[1]:
                    outputs = outputs[:, :original_shape[1]]
                
                # Save tensors for backward pass
                if clipping_mode == "learnable":
                    ctx.save_for_backward(inputs, scale, zero_point, beta, gamma, inputs_q)
                else:
                    ctx.save_for_backward(inputs, scale, zero_point, None, None, inputs_q)
                return outputs
            else:
                # For other tensor shapes, fall back to per-tensor quantization
                group_size = -1
        
        # Per-channel quantization
        if per_channel and group_size < 0:
            # Find min/max per channel
            if channel_dim == 0:
                min_val = inputs.min(dim=1, keepdim=True)[0]
                max_val = inputs.max(dim=1, keepdim=True)[0]
            else:
                min_val = inputs.min(dim=0, keepdim=True)[0]
                max_val = inputs.max(dim=0, keepdim=True)[0]
                
            # Apply clipping if needed
            if clipping_mode == "fixed_ratio":
                min_val = min_val * clipping_ratio
                max_val = max_val * clipping_ratio
            elif clipping_mode == "learnable" and beta is not None and gamma is not None:
                min_val = torch.sigmoid(beta) * min_val
                max_val = torch.sigmoid(gamma) * max_val
            
            # Compute scale and zero point
            qmin = 0
            qmax = 2**bits - 1
            scale = (max_val - min_val) / qmax
            
            # Add small epsilon to avoid division by zero
            scale = scale + 1e-8
            
            zero_point = qmin - min_val / scale
            zero_point = torch.round(zero_point)
            zero_point = torch.clamp(zero_point, qmin, qmax)
            
            # Manual quantization (matching fake_quantize_state_dict behavior)
            inputs_q = torch.round(inputs / scale + zero_point)
            inputs_q = torch.clamp(inputs_q, qmin, qmax)
            outputs = (inputs_q - zero_point) * scale
            
        # Per-tensor quantization
        elif group_size < 0:
            # Global min/max
            min_val = inputs.min()
            max_val = inputs.max()
            
            # Apply clipping if needed
            if clipping_mode == "fixed_ratio":
                min_val = min_val * clipping_ratio
                max_val = max_val * clipping_ratio
            elif clipping_mode == "learnable" and beta is not None and gamma is not None:
                min_val = torch.sigmoid(beta) * min_val
                max_val = torch.sigmoid(gamma) * max_val
            
            # Compute scale and zero point
            qmin = 0
            qmax = 2**bits - 1
            scale = (max_val - min_val) / qmax
            
            # Add small epsilon (matching fake_quantize_state_dict)
            scale = scale + 1e-8
            
            zero_point = qmin - min_val / scale
            zero_point = torch.round(zero_point)
            zero_point = torch.clamp(zero_point, qmin, qmax)
            
            # Manual quantization to match fake_quantize_state_dict
            inputs_q = torch.round(inputs / scale + zero_point)
            inputs_q = torch.clamp(inputs_q, qmin, qmax)
            outputs = (inputs_q - zero_point) * scale
            
        # Save tensors for backward pass
        if clipping_mode == "learnable":
            ctx.save_for_backward(inputs, scale, zero_point, beta, gamma, inputs_q)
        else:
            ctx.save_for_backward(inputs, scale, zero_point, None, None, inputs_q)
        
        return outputs
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Straight-through estimator for backward pass with correctly shaped gradients for learnable clipping parameters
        """
        saved_tensors = ctx.saved_tensors
        inputs = saved_tensors[0]
        scale = saved_tensors[1]
        zero_point = saved_tensors[2]
        
        # Get learnable parameters if available
        beta = gamma = None
        if ctx.clipping_mode == "learnable" and len(saved_tensors) > 3:
            beta = saved_tensors[3]
            gamma = saved_tensors[4]
            inputs_q = saved_tensors[5] if len(saved_tensors) > 5 else None
        else:
            inputs_q = saved_tensors[3] if len(saved_tensors) > 3 else None
        
        # Mask for gradients of values that would be clipped in quantization
        qmin = 0
        qmax = 2**ctx.bits - 1
        
        # Initialize gradients for optional parameters
        beta_grad = gamma_grad = None
        
        # Use STE (Straight-Through Estimator) for the gradient of the weights
        grad_input = grad_output.clone()
        
        if ctx.clipping_mode == "learnable" and beta is not None and gamma is not None:
            # For group quantization
            if ctx.group_size > 0:
                # Shapes must match exactly
                original_shape = inputs.shape
                
                if len(original_shape) == 3:  # Already reshaped for group quantization
                    out_channels, num_groups, group_size = original_shape
                    
                    # Compute per-group min/max values
                    min_val = inputs.min(dim=2, keepdim=True)[0]  # Shape: [out_channels, num_groups, 1]
                    max_val = inputs.max(dim=2, keepdim=True)[0]  # Shape: [out_channels, num_groups, 1]
                    
                    # Calculate gradients for beta and gamma with correct shapes
                    sigmoid_beta = torch.sigmoid(beta)  # Shape should match beta
                    dsigmoid_beta = sigmoid_beta * (1 - sigmoid_beta)
                    
                    sigmoid_gamma = torch.sigmoid(gamma)  # Shape should match gamma
                    dsigmoid_gamma = sigmoid_gamma * (1 - sigmoid_gamma)
                    
                    # Reshape grad_output to match the groups
                    grad_output_reshaped = grad_output.reshape(out_channels, num_groups, group_size)
                    
                    # Sum gradients over the group dimension (dim=2) to match beta/gamma shape
                    beta_grad = torch.sum(grad_output_reshaped, dim=2, keepdim=True) * dsigmoid_beta * min_val
                    gamma_grad = torch.sum(grad_output_reshaped, dim=2, keepdim=True) * dsigmoid_gamma * max_val
                    
                    # Remove the keepdim dimension to match beta/gamma shape exactly
                    beta_grad = beta_grad.squeeze(2)  # Should be [out_channels, num_groups]
                    gamma_grad = gamma_grad.squeeze(2)  # Should be [out_channels, num_groups]
                    
                else:
                    # Handle case where inputs shape is [out_channels, in_features]
                    out_channels, in_features = original_shape
                    num_groups = in_features // ctx.group_size
                    
                    # Reshape for proper gradient calculation
                    inputs_reshaped = inputs.reshape(out_channels, num_groups, ctx.group_size)
                    grad_output_reshaped = grad_output.reshape(out_channels, num_groups, ctx.group_size)
                    
                    min_val = inputs_reshaped.min(dim=2, keepdim=True)[0]
                    max_val = inputs_reshaped.max(dim=2, keepdim=True)[0]
                    
                    sigmoid_beta = torch.sigmoid(beta)
                    dsigmoid_beta = sigmoid_beta * (1 - sigmoid_beta)
                    
                    sigmoid_gamma = torch.sigmoid(gamma)
                    dsigmoid_gamma = sigmoid_gamma * (1 - sigmoid_gamma)
                    
                    # Sum over group dimension (dim=2)
                    beta_grad = torch.sum(grad_output_reshaped, dim=2, keepdim=True) * dsigmoid_beta * min_val
                    gamma_grad = torch.sum(grad_output_reshaped, dim=2, keepdim=True) * dsigmoid_gamma * max_val
                    
                    # Remove keepdim to match beta/gamma shape
                    beta_grad = beta_grad.squeeze(2)
                    gamma_grad = gamma_grad.squeeze(2)
                    
            elif ctx.per_channel:
                # Per-channel quantization - gradients must match the shape of beta/gamma
                if ctx.channel_dim == 0:
                    # Get min/max along dim=1 (keeping dim=0 which is the channel dimension)
                    min_val = inputs.min(dim=1, keepdim=True)[0]  # Shape: [out_channels, 1]
                    max_val = inputs.max(dim=1, keepdim=True)[0]  # Shape: [out_channels, 1]
                    
                    sigmoid_beta = torch.sigmoid(beta)
                    dsigmoid_beta = sigmoid_beta * (1 - sigmoid_beta)
                    
                    sigmoid_gamma = torch.sigmoid(gamma)
                    dsigmoid_gamma = sigmoid_gamma * (1 - sigmoid_gamma)
                    
                    # Sum gradients along dim=1 to match beta/gamma shape
                    beta_grad = torch.sum(grad_output, dim=1, keepdim=True) * dsigmoid_beta * min_val
                    gamma_grad = torch.sum(grad_output, dim=1, keepdim=True) * dsigmoid_gamma * max_val
                    
                    # Ensure shape exactly matches beta and gamma
                    if beta.shape != beta_grad.shape:
                        beta_grad = beta_grad.reshape(beta.shape)
                        gamma_grad = gamma_grad.reshape(gamma.shape)
                else:
                    # Handle case where channel_dim is not 0
                    min_val = inputs.min(dim=0, keepdim=True)[0]
                    max_val = inputs.max(dim=0, keepdim=True)[0]
                    
                    sigmoid_beta = torch.sigmoid(beta)
                    dsigmoid_beta = sigmoid_beta * (1 - sigmoid_beta)
                    
                    sigmoid_gamma = torch.sigmoid(gamma)
                    dsigmoid_gamma = sigmoid_gamma * (1 - sigmoid_gamma)
                    
                    # Sum gradients along dim=0 to match beta/gamma shape
                    beta_grad = torch.sum(grad_output, dim=0, keepdim=True) * dsigmoid_beta * min_val
                    gamma_grad = torch.sum(grad_output, dim=0, keepdim=True) * dsigmoid_gamma * max_val
                    
                    # Ensure shape exactly matches beta and gamma
                    if beta.shape != beta_grad.shape:
                        beta_grad = beta_grad.reshape(beta.shape)
                        gamma_grad = gamma_grad.reshape(gamma.shape)
            else:
                # Per-tensor quantization - beta and gamma are scalars
                min_val = inputs.min()
                max_val = inputs.max()
                
                sigmoid_beta = torch.sigmoid(beta)
                dsigmoid_beta = sigmoid_beta * (1 - sigmoid_beta)
                
                sigmoid_gamma = torch.sigmoid(gamma)
                dsigmoid_gamma = sigmoid_gamma * (1 - sigmoid_gamma)
                
                # For scalar beta/gamma, shape is just [1]
                beta_grad = torch.sum(grad_output).reshape_as(beta) * dsigmoid_beta * min_val
                gamma_grad = torch.sum(grad_output).reshape_as(gamma) * dsigmoid_gamma * max_val
        
        # Return gradients for all input arguments
        # Most will be None since they are not torch.Tensors requiring gradients
        return_values = [grad_input, None, None, None, None, None, None, None, None, None]
        
        # Add gradients for learnable parameters if used
        if ctx.clipping_mode == "learnable":
            return_values.extend([beta_grad, gamma_grad])
        else:
            return_values.extend([None, None])
            
        return tuple(return_values)


class QuantizedLinear(nn.Module):
    def __init__(self, linear_layer, bits=8, per_channel=False, group_size=-1, 
                 clipping_mode="none", clipping_ratio=1.0, beta_init=0.0, gamma_init=0.0):
        # Removed stochastic_quantization parameter
        super().__init__()
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        
        # Copy the weights and biases
        self.weight = nn.Parameter(linear_layer.weight.data)
        self.bias = nn.Parameter(linear_layer.bias.data) if linear_layer.bias is not None else None
        
        # Quantization parameters
        self.bits = bits
        self.per_channel = per_channel
        self.group_size = group_size
        self.quantizer = AsymmetricQuantizer.apply
        
        # Clipping parameters
        self.clipping_mode = clipping_mode
        self.clipping_ratio = clipping_ratio
        
        # Initialize learnable clipping parameters with correct dimensions if needed
        if clipping_mode == "learnable":
            if group_size > 0:
                # For group quantization, we need parameters per group
                # Calculate number of groups
                num_groups = self.in_features // group_size
                if self.in_features % group_size != 0:
                    num_groups += 1  # Account for padding
                
                # Initialize parameters with shape [out_features, num_groups]
                self.beta = nn.Parameter(torch.ones(self.out_features, num_groups) * beta_init)
                self.gamma = nn.Parameter(torch.ones(self.out_features, num_groups) * gamma_init)
            elif per_channel:
                # For per-channel quantization, we need parameters per output channel
                self.beta = nn.Parameter(torch.ones(self.out_features, 1) * beta_init)
                self.gamma = nn.Parameter(torch.ones(self.out_features, 1) * gamma_init)
            else:
                # For per-tensor quantization, scalar parameters are sufficient
                self.beta = nn.Parameter(torch.ones(1) * beta_init)
                self.gamma = nn.Parameter(torch.ones(1) * gamma_init)
        else:
            self.register_buffer('beta', None)
            self.register_buffer('gamma', None)

    def forward(self, input):
        """
        Forward pass with operation-specific quantization control.
        
        Args:
            input: Input tensor
            op_type: Current operation type ("ACTOR_UPDATE", "LOG_PROB", "ROLLOUT")
                     If None, uses default behavior
        """
        
        # Always apply quantization (removed stochastic logic)
        apply_quant = True

        if apply_quant:
            # Fake quantize the weights during forward pass
            if self.group_size > 0:
                # Apply group quantization
                quantized_weight = self.quantizer(
                    self.weight, 
                    self.bits, 
                    None, 
                    None, 
                    False,  # per_channel not used with group quantization
                    0,      # channel_dim not used with group quantization
                    self.group_size, 
                    1,      # group along in_features dimension
                    self.clipping_mode,
                    self.clipping_ratio,
                    self.beta if self.clipping_mode == "learnable" else None,
                    self.gamma if self.clipping_mode == "learnable" else None
                )
            else:
                # Apply per-tensor or per-channel quantization
                quantized_weight = self.quantizer(
                    self.weight, 
                    self.bits, 
                    None, 
                    None, 
                    self.per_channel, 
                    0 if self.per_channel else None,
                    -1,
                    1,
                    self.clipping_mode,
                    self.clipping_ratio,
                    self.beta if self.clipping_mode == "learnable" else None,
                    self.gamma if self.clipping_mode == "learnable" else None
                )
            
            # Use the quantized weights for the forward pass
            return F.linear(input, quantized_weight, self.bias)
        else:
            # Use original weights
            return F.linear(input, self.weight, self.bias)


def quantize_model_weights(model, bits=8, exclude_modules=None, per_channel=False, group_size=-1, 
                          clipping_mode="none", clipping_ratio=1.0, beta_init=0.0, gamma_init=0.0):
    """
    Apply quantization to model weights (excluding specified modules).
    
    Args:
        model: The model to quantize
        bits: Number of bits for quantization (default: 8)
        exclude_modules: List of module names to exclude from quantization
        per_channel: Whether to use per-channel quantization
        group_size: Size of groups for group quantization (-1 disables group quantization)
        clipping_mode: Weight clipping mode ("none", "fixed_ratio", or "learnable")
        clipping_ratio: Ratio for fixed_ratio clipping mode
        beta_init: Initial value for learnable beta (min clipping) parameter
        gamma_init: Initial value for learnable gamma (max clipping) parameter
        
    Returns:
        Modified model with quantized linear layers
    """
    if exclude_modules is None:
        exclude_modules = []
        
    # Add default excluded layers (embeddings and final layer)
    default_excludes = ['embed_tokens', 'lm_head', 'reward_head']
    exclude_modules.extend(default_excludes)
    
    # Function to check if a module should be excluded
    def should_exclude(name):
        return any(excluded in name for excluded in exclude_modules)

    # Count quantized layers
    quantized_count = 0
    
    # Replace linear layers with quantized versions
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and not should_exclude(name):
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            
            if parent_name:
                parent = model
                for part in parent_name.split('.'):
                    parent = getattr(parent, part)
                    
                # Replace the linear layer with a quantized version - removed stochastic_quantization
                setattr(parent, child_name, QuantizedLinear(
                    module, 
                    bits=bits, 
                    per_channel=per_channel,
                    group_size=group_size,
                    clipping_mode=clipping_mode,
                    clipping_ratio=clipping_ratio,
                    beta_init=beta_init,
                    gamma_init=gamma_init
                ))
                quantized_count += 1
            else:
                # Handle case where module is a direct child of the model - removed stochastic_quantization
                setattr(model, child_name, QuantizedLinear(
                    module, 
                    bits=bits, 
                    per_channel=per_channel,
                    group_size=group_size,
                    clipping_mode=clipping_mode,
                    clipping_ratio=clipping_ratio,
                    beta_init=beta_init,
                    gamma_init=gamma_init
                ))
                quantized_count += 1
    return model