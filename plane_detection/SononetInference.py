import base64
import os
import cv2
import onnx
import onnxruntime
import numpy as np
from scipy import stats

from loguru import logger


class SononetInference:
    def __init__(self):
        self.model_path = f"{os.path.dirname(__file__)}/sononet.onnx"

        self.class_names = ['3VT', '4CH', 'ABDOMINAL', 'BACKGROUND', 'BRAIN-CB', 'BRAIN-TV', 'FEMUR', 'KIDNEYS', 'LIPS',
                            'LVOT', 'PROFILE', 'RVOT', 'SPINE-CORONAL', 'SPINE-SAGITTAL']

        self.background_index = self.class_names.index('BACKGROUND')

        self.output_layers = ['351', 'x', 'onnx::AveragePool_343']

        self.class_buffers = [[] for _ in self.class_names]
        self.blur_buffers = [[] for _ in self.class_names]
        self.class_probabilities_buffer = [[] for _ in self.class_names]
        self.max_buffer_size = [200 for _ in self.class_names]

        self.class_indices = [i for i, _ in enumerate(self.class_names)]
        self.class_name_map = dict(zip(self.class_names, self.class_indices))
        self.class_probs = [0 for _ in self.class_names]
        self.rolling_window_size = 60
        self.rolling_winner_window = [3 for _ in range(self.rolling_window_size)]
        self.class_alphas = [0 for _ in self.class_names]
        self.threshold_background_reduction = 1
        self.threshold_winner_label = [0.75 for _ in self.class_names] 
        self.winner = 'AWAITING STREAM'
        model = self.check_model(self.model_path)
        self.ort_session = onnxruntime.InferenceSession(model.SerializeToString(), providers=['CUDAExecutionProvider'])
        logger.info(f"Running on device: {onnxruntime.get_device()}")

    def check_model(self, path: str):

        model = onnx.load(path)
        inter_layers = self.output_layers
        value_info_protos = []
        shape_info = onnx.shape_inference.infer_shapes(model)

        for idx, node in enumerate(shape_info.graph.value_info):
            if node.name in inter_layers:
                value_info_protos.append(node)

        model.graph.output.extend(value_info_protos)
        onnx.checker.check_model(model)
    
        return model

    def detect_scan_planes(self, arr):
        frame = arr / 255
        frame = np.expand_dims(np.expand_dims(cv2.resize(frame, dsize=(288, 224)), 0), 0).astype(np.float32)
        ort_inputs = {self.ort_session.get_inputs()[0].name: frame}
        outs = self.ort_session.run(self.output_layers, ort_inputs)

        features = outs[1][0].tolist()
        full_saliency = outs[2][0]
        outs = np.array(outs[:2])[0][0].astype(float)

        # outs[self.background_index] = outs[self.background_index] * self.threshold_background_reduction
        pred = np.argmax(outs, axis=0)
        class_saliency = full_saliency[pred]

        self.class_probs = outs
        self.winner = self.class_names[pred]

        # return {'probability': outs[pred], 'label': self.winner, 'features': features, 'full_saliency': full_saliency, 'class_saliency': class_saliency}
        return outs[pred], self.winner, class_saliency
    
    @staticmethod
    def crop_frame(image):
        offset_r = int(image.shape[0] * 0.22)
        offset_c = int(image.shape[1] * 0.15)
        image = image[
                offset_r:image.shape[0] - int(image.shape[0] * 0.22),
                offset_c:image.shape[1] - int(image.shape[1] * 0.15)
                ]
        return image
