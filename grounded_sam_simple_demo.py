import cv2
import numpy as np
import supervision as sv
import os 

import torch
import torchvision

from GroundingDINO.groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor

class GroundingSAM:
    def __init__(self):  
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.grounding_dino_config_path = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        self.grounding_dino_checkpoint_path = "ckpt/groundingdino_swint_ogc.pth"

        # Segment-Anything checkpoint
        self.sam_encoder_version = "vit_h"
        self.sam_checkpoint_path = "ckpt/sam_vit_h_4b8939.pth"

        self.grounding_dino_model = Model(model_config_path=self.grounding_dino_config_path, model_checkpoint_path=self.grounding_dino_checkpoint_path)

        # Building SAM Model and SAM Predictor
        sam = sam_model_registry[self.sam_encoder_version](checkpoint=self.sam_checkpoint_path)
        sam.to(device=self.device)
        self.sam_predictor = SamPredictor(sam)
    
    def segment(self, sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
        sam_predictor.set_image(image)
        result_masks = []
        for box in xyxy:
            masks, scores, logits = sam_predictor.predict(
                box=box,
                multimask_output=True
            )
            index = np.argmax(scores)
            result_masks.append(masks[index])
        return np.array(result_masks)

    def __call__(self, image, classes , box_threshold=0.25, text_threshold=0.25, nms_threshold=0.8):
        detections = self.grounding_dino_model.predict_with_classes(
                image=image,
                classes=classes,
                box_threshold=box_threshold,
                text_threshold=text_threshold
        )
        box_annotator = sv.BoxAnnotator()
        labels = [
            f"{classes[class_id]} {confidence:0.2f}" 
            for _, _, confidence, class_id, _, _ 
            in detections]
        annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)
        # cv2.imwrite("output/groundingdino_annotated_image.jpg", annotated_frame)
        # NMS post process
        nms_idx = torchvision.ops.nms(
            torch.from_numpy(detections.xyxy), 
            torch.from_numpy(detections.confidence), 
            nms_threshold
        ).numpy().tolist()
        detections.xyxy = detections.xyxy[nms_idx]
        detections.confidence = detections.confidence[nms_idx]
        detections.class_id = detections.class_id[nms_idx]
        mask = detections.mask = self.segment(
            sam_predictor=self.sam_predictor,
            image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            xyxy=detections.xyxy
        )
        
        mask = np.any(mask, axis=0)
        mask = mask.astype(int)
        mask = np.where(mask == 1, 255, 0)

        box_annotator = sv.BoxAnnotator()
        mask_annotator = sv.MaskAnnotator()
        labels = [
            f"{classes[class_id]} {confidence:0.2f}" 
            for _, _, confidence, class_id, _, _ 
            in detections]
        annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
        annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
        return mask, annotated_image, annotated_frame
        # cv2.imwrite("output/grounded_sam_annotated_image.jpg", annotated_image)


if __name__ == "__main__":

    # load image
    image = cv2.imread("./assets/demo1.jpg")
    classes = ["a running dog"]
    groundingsam = GroundingSAM()
    mask,_,_ = groundingsam(image, classes)

