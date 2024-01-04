from pydantic import BaseModel
from transformers import YolosForObjectDetection, YolosImageProcessor
from PIL import Image
import torch

class Object(BaseModel):
    box: tuple[float, float, float, float]
    label: str

class Objects(BaseModel):
    objects: list[Object]

class ObjectDetection:
    # Attribute to hold an instance of YolosImageProcessor, initialized to None
    image_processor: YolosImageProcessor | None = None
    
    # Attribute to hold an instance of YolosForObjectDetection, initialized to None
    model: YolosForObjectDetection | None = None

    def load_model(self) -> None:
        """Load the model"""
        # Initialize the image processor with a pretrained model
        self.image_processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")
        
        # Initialize the object detection model with a pretrained model
        self.model = YolosForObjectDetection.from_pretrained("hustvl/yolos-tiny")

    def predict(self, image: Image.Image) -> Objects:
        """Runs a prediction"""
        # Check if the image processor or model is not loaded
        if not self.image_processor or not self.model:
            # Raise an error if the model is not loaded
            raise RuntimeError("Model is not loaded")
        
        # Preprocess the image using the image processor, converting it into tensors
        inputs = self.image_processor(images=image, return_tensors="pt")
        
        # Pass the preprocessed inputs through the object detection model to get outputs
        outputs = self.model(**inputs)

        # Convert the image size to a tensor
        target_sizes = torch.tensor([image.size[::-1]])
        
        # Process the object detection outputs using the image processor
        results = self.image_processor.post_process_object_detection(
            outputs, target_sizes=target_sizes
        )[0]

        # Create an empty list to store detected objects
        objects: list[Object] = []

        # Iterate through the detected results: scores, labels, and bounding boxes
        for score, label, box in zip(
            results["scores"], results["labels"], results["boxes"]
        ):
            # Check if the score for the detected object is above a threshold (0.7)
            if score > 0.7:
                # Convert bounding box coordinates to a list
                box_values = box.tolist()
                
                # Retrieve the label name based on the label's ID using the model's configuration
                label = self.model.config.id2label[label.item()]
                
                # Create an Object instance and append it to the objects list
                objects.append(Object(box=box_values, label=label))
        
        # Return the detected objects as an Objects object
        return Objects(objects=objects)

# Create an instance of the ObjectDetection class
object_detection = ObjectDetection()
