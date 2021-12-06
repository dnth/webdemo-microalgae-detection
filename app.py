from gradio.outputs import Label
from icevision.all import *
from icevision.models.checkpoint import *
import PIL
import gradio as gr
import os

# Load model
checkpoint_path = "models/model_checkpoint.pth"
checkpoint_and_model = model_from_checkpoint(checkpoint_path)

model = checkpoint_and_model["model"]
model_type = checkpoint_and_model["model_type"]
class_map = checkpoint_and_model["class_map"]

# Transforms
img_size = checkpoint_and_model["img_size"]
valid_tfms = tfms.A.Adapter([*tfms.A.resize_and_pad(img_size), tfms.A.Normalize()])


for root, dirs, files in os.walk(r"sample_images/"):
    for filename in files:
        print(filename)

examples = ["sample_images/" + file for file in files]
article = "<p style='text-align: center'><a href='https://dicksonneoh.com/' target='_blank'>Blog post</a></p>"
enable_queue = True

# Populate examples in Gradio interface
example_images = [["sample_images/" + file] for file in files]

# Columns: Input Image | Label | Box | Detection Threshold
examples = [
    [example_images[0], False, True, 0.5],
    [example_images[1], True, True, 0.5],
    [example_images[2], False, True, 0.7],
    [example_images[3], True, True, 0.7],
    [example_images[4], False, True, 0.5],
    [example_images[5], False, True, 0.5],
    [example_images[6], False, True, 0.5],
    [example_images[7], False, True, 0.5],
]

def show_preds(input_image, display_label, display_bbox, detection_threshold):

    if detection_threshold == 0:
        detection_threshold = 0.5

    img = PIL.Image.fromarray(input_image, "RGB")

    pred_dict = model_type.end2end_detect(
        img,
        valid_tfms,
        model,
        class_map=class_map,
        detection_threshold=detection_threshold,
        display_label=display_label,
        display_bbox=display_bbox,
        return_img=True,
        font_size=16,
        label_color="#FF59D6",
    )

    return pred_dict["img"], len(pred_dict["detection"]["bboxes"])


# display_chkbox = gr.inputs.CheckboxGroup(["Label", "BBox"], label="Display", default=True)
display_chkbox_label = gr.inputs.Checkbox(label="Label", default=False)
display_chkbox_box = gr.inputs.Checkbox(label="Box", default=True)

detection_threshold_slider = gr.inputs.Slider(
    minimum=0, maximum=1, step=0.1, default=0.5, label="Detection Threshold"
)

outputs = [
    gr.outputs.Image(type="pil", label="RetinaNet Inference"),
    gr.outputs.Textbox(type='number', label='Microalgae Count')
    ]

# Option 1: Get an image from local drive
gr_interface = gr.Interface(
    fn=show_preds,
    inputs=[
        "image",
        display_chkbox_label,
        display_chkbox_box,
        detection_threshold_slider,
    ],
    outputs=outputs,
    title="Microalgae Detector with RetinaNet",
    description="This RetinaNet model counts microalgaes on a given image. Upload an image or click an example image below to use.",
    article=article,
    examples=examples,
)


# #  Option 2: Grab an image from a webcam
# gr_interface = gr.Interface(fn=show_preds, inputs=["webcam", display_chkbox_label, display_chkbox_box,  detection_threshold_slider], outputs=outputs, title='IceApp - COCO', live=False)

# #  Option 3: Continuous image stream from the webcam
# gr_interface = gr.Interface(fn=show_preds, inputs=["webcam", display_chkbox_label, display_chkbox_box,  detection_threshold_slider], outputs=outputs, title='IceApp - COCO', live=True)


gr_interface.launch(inline=False, share=True, debug=True)
