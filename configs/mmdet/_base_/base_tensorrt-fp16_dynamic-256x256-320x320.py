_base_ = ['./base_dynamic.py', '../../_base_/backends/tensorrt-fp16.py']

backend_config = dict(
    common_config=dict(max_workspace_size=1 << 30),
    model_inputs=[
        dict(
            input_shapes=dict(
                input=dict(
                    min_shape=[1, 3, 256, 256],
                    opt_shape=[1, 3, 256, 256],
                    max_shape=[1, 3, 320, 320])))
    ])
