_base_ = ['./segmentation_static.py', '../_base_/backends/tensorrt-int8.py']

onnx_config = dict(input_shape=[848, 480])
backend_config = dict(
    common_config=dict(max_workspace_size=1 << 30),
    model_inputs=[
        dict(
            input_shapes=dict(
                input=dict(
                    min_shape=[1, 3, 480, 848],
                    opt_shape=[1, 3, 480, 848],
                    max_shape=[1, 3, 480, 848])))
    ])
