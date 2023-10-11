import cv2
import numpy as np
import os


def find_bounding_box(image):
    """
    Finds the bounding box of the non-white object in the image.
    Assumes the image has a white background.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Use a slightly lower threshold value
    _, thresh = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY_INV)
    
    # Find all contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # If no contours are detected, return the whole image as bounding box
    if not contours:
        return 0, 0, image.shape[1], image.shape[0]
    
    # Consider all detected contours to determine the bounding box
    x_vals = []
    y_vals = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x_vals.extend([x, x+w])
        y_vals.extend([y, y+h])
    
    x_min, x_max = min(x_vals), max(x_vals)
    y_min, y_max = min(y_vals), max(y_vals)
    
    return x_min, y_min, x_max - x_min, y_max - y_min


def compute_padding(cropped_shape, template_size):
    """
    Computes the padding required for the cropped image so that after resizing,
    it will fit within the template size with a 200-pixel white border.
    """
    effective_width = template_size[0] - 400
    effective_height = template_size[1] - 400
    width_scale = effective_width / cropped_shape[1]
    height_scale = effective_height / cropped_shape[0]
    scale = min(width_scale, height_scale)
    resized_width = int(cropped_shape[1] * scale)
    resized_height = int(cropped_shape[0] * scale)
    pad_width_total = template_size[0] - resized_width
    pad_height_total = template_size[1] - resized_height
    pad_left = pad_width_total // 2
    pad_right = pad_width_total - pad_left
    pad_top = pad_height_total // 2
    pad_bottom = pad_height_total - pad_top
    return pad_top, pad_bottom, pad_left, pad_right

# Update the batch processing function to use the refined bounding box function

def resize_and_pad(image, template_size):
    """Resize the image while preserving its aspect ratio to fit within the effective area of the template.
    Then, pad the image to match the template size."""
    effective_width = template_size[0] - 400
    effective_height = template_size[1] - 400
    image_aspect = image.shape[1] / image.shape[0]
    effective_aspect = effective_width / effective_height
    if image_aspect > effective_aspect:
        new_width = effective_width
        new_height = int(effective_width / image_aspect)
    else:
        new_height = effective_height
        new_width = int(effective_height * image_aspect)


        # Before the cv2.resize call:
    print(f"Calculated new dimensions: {new_width}x{new_height}")  # Debug print
    if new_width <= 0 or new_height <= 0:
        print(f"Error: Invalid calculated dimensions for resizing: {new_width}x{new_height}")
        return None  # or handle this error appropriately
    
    resized_image = cv2.resize(image, (new_width, new_height))

    resized_image = cv2.resize(image, (new_width, new_height))
    pad_top = (template_size[1] - new_height) // 2
    pad_bottom = template_size[1] - new_height - pad_top
    pad_left = (template_size[0] - new_width) // 2
    pad_right = template_size[0] - new_width - pad_left
    padded_image = cv2.copyMakeBorder(resized_image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    return padded_image


def batch_resize(input_dir, output_dir, boxed_output_dir, template_size):
    """
    Processes all images in the input directory and resizes them based on the provided steps.
    Also saves images with bounding boxes drawn.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(boxed_output_dir):
        os.makedirs(boxed_output_dir)

    for image_name in os.listdir(input_dir):
        if not (image_name.endswith(".jpg") or image_name.endswith(".png")):
            continue
        image_path = os.path.join(input_dir, image_name)
        image = cv2.imread(image_path)
        
        # Find bounding box and draw on the original image
        x, y, w, h = find_bounding_box(image)
        boxed_image = cv2.rectangle(image.copy(), (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Save the boxed image
        boxed_output_path = os.path.join(boxed_output_dir, image_name)
        cv2.imwrite(boxed_output_path, boxed_image)
        
        # Continue with the resizing process
        cropped_image = image[y:y+h, x:x+w]
        padding = compute_padding(cropped_image.shape[:2], template_size)
        padded_image = cv2.copyMakeBorder(cropped_image, padding[0], padding[1], padding[2], padding[3], cv2.BORDER_CONSTANT, value=[255, 255, 255])
        final_image = resize_and_pad(cropped_image, template_size)
        output_path = os.path.join(output_dir, image_name)
        cv2.imwrite(output_path, final_image)



if __name__ == "__main__":
    input_dir = "raw"
    output_dir = "resized"
    # batch_resize("inputs", "outputs", (1000, 1000))  # For 1000x1000 template
    # batch_resize("inputs", "outputs", (1801, 2600))  # For 1801x2600 template


    # updated function on a problematic image
    # problematic_image_path = os.path.join(input_dir, "sneakers2")  # Adjust the extension if needed
    # if os.path.exists(problematic_image_path):
    #     problematic_image = cv2.imread(problematic_image_path)
    #     bbox_updated = find_bounding_box(problematic_image)
    #     cropped_image_updated = problematic_image[bbox_updated[1]:bbox_updated[1]+bbox_updated[3], bbox_updated[0]:bbox_updated[0]+bbox_updated[2]]
    #     cv2.imwrite(os.path.join(output_dir, "cropped_image_updated.png"), cropped_image_updated)
    # else:
    #     print(f"Image {problematic_image_path} not found!")


    if os.path.exists(input_dir):
        boxed_output_dir = "boxed_outputs"
        template_size_sample = (1000, 1000)
        batch_resize(input_dir, output_dir, boxed_output_dir, template_size_sample)
        
    # Check if images with bounding boxes were saved
    boxed_images = os.listdir(boxed_output_dir) if os.path.exists(boxed_output_dir) else []
    boxed_images


