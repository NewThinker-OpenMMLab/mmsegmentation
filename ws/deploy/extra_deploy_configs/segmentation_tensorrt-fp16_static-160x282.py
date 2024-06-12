_base_ = ['./segmentation_static.py', '../_base_/backends/tensorrt-fp16.py']

onnx_config = dict(input_shape=[282, 160])
backend_config = dict(
    common_config=dict(max_workspace_size=1 << 30),
    model_inputs=[
        dict(
            input_shapes=dict(
                input=dict(
                    min_shape=[1, 3, 160, 282],
                    opt_shape=[1, 3, 160, 282],
                    max_shape=[1, 3, 160, 282])))
    ])
