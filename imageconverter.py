import cv2
import os

input_folder = "student_images"
output_folder = "student_images_rgb"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Loop through each file
for filename in os.listdir(input_folder):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        img_path = os.path.join(input_folder, filename)
        image = cv2.imread(img_path)

        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Save as .png or same format
        save_path = os.path.join(output_folder, filename)
        cv2.imwrite(save_path, cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))  # Saving still in OpenCV-friendly format
        print(f"[✅] Converted: {filename}")

print("\n[🎉] All images converted to RGB format successfully!")
