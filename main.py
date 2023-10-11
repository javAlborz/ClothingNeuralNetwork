import argparse
import cv2
import image_resizer
import inference

def main(image_path, skip_resize):
    # Load the original image
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"Error: Could not read image at {image_path}")
        return
    
    if not skip_resize:
        # Resize and pad the image
        resized_image = image_resizer.resize_and_pad(original_image, (1000, 1000))
        # You may need to save the resized image to disk temporarily if your inference function requires a path
        temp_resized_path = "temp_resized_image.jpg"
        cv2.imwrite(temp_resized_path, resized_image)
        inference_image_path = temp_resized_path
    else:
        inference_image_path = image_path
    
    # Run inference and plotting on the image using functions from inference.py
    result = inference.run_inference_and_plot(inference_image_path)  # Pass the appropriate image path
    print(result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run inference on an image.')
    parser.add_argument('image_path', help='Path to the input image.')
    parser.add_argument('--skip_resize', action='store_true', help='Skip image resizing step if this flag is present.')
    args = parser.parse_args()
    main(args.image_path, args.skip_resize)
