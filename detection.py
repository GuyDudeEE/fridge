from inference import get_model
import supervision as sv
import cv2
import re

# define the image url to use for inference
image_file = "people-walking.jpg"
image = cv2.imread(image_file)

# load a pre-trained yolov8n model
#model = get_model(model_id="groceries-tbavo/1")
model = get_model(model_id="yolov8n-640")

# run inference on our chosen image, image can be a url, a numpy array, a PIL image, etc.
results = model.infer(image)[0]

# load the results into the supervision Detections api
detections = sv.Detections.from_inference(results)

# create supervision annotators
bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

# annotate the image with our inference results
annotated_image = bounding_box_annotator.annotate(
    scene=image, detections=detections)
annotated_image = label_annotator.annotate(
    scene=annotated_image, detections=detections)

#Obtain plain text of what was found on screen
infs = str(detections)
pattern = r"'([^']*)"
matches = re.findall(pattern, infs)
guess_list = ' '.join(matches)
patt2=r'\b\w+\b'
match2 = re.findall(patt2, guess_list)
actual = ' '.join(match2)
word_list=actual.split()
plz = word_list[3:-2]
objects_on_screen= ' '.join(plz)
print(objects_on_screen)

# display the image
#print(detections)
#sv.plot_image(annotated_image)