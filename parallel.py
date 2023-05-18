import os
import cv2
import sys
import multiprocessing
import time

# Path to the folder containing training images
training_folder_path = "C:/Users/Rameesha/Desktop/TrainingImage"

# Load the face detection model
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load all the training images and their names into a dictionary
training_images = {}
for file_name in os.listdir(training_folder_path):
    if file_name.endswith(".jpg") or file_name.endswith(".jpeg") or file_name.endswith(".png"):
        image_path = os.path.join(training_folder_path, file_name)
        training_images[file_name] = cv2.imread(image_path)

def calculate_histogram_similarity(face_box, gray_input_image):
    x, y, w, h = face_box
    face_roi = gray_input_image[y:y+h, x:x+w]

    # Resize the face ROI to the same size as the training images
    resized_face_roi = cv2.resize(face_roi, (100, 100))

    best_match_name = ""
    best_match_score = 0

    # Iterate through each training image and compare their histogram similarity to the input face
    for name, training_image in training_images.items():
        resized_training_image = cv2.resize(training_image, (100, 100))
        score = cv2.compareHist(cv2.calcHist([resized_face_roi], [0], None, [256], [0, 256]),
                                cv2.calcHist([resized_training_image], [0], None, [256], [0, 256]),
                                cv2.HISTCMP_CORREL)
        if score > best_match_score:
            best_match_name = name
            best_match_score = score

    return best_match_name

def main():
    # Get the user input image path
    input_image_path = input("Enter the path to the input image: ")
 # Start measuring execution time
    start_time = time.time()
    # Check if the input image file exists
    if not os.path.isfile(input_image_path):
        print("Invalid input image path!")
        sys.exit(1)

    # Load the input image
    input_image = cv2.imread(input_image_path)

    # Detect faces in the input image
    gray_input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    detected_faces = face_detector.detectMultiScale(gray_input_image, scaleFactor=1.1, minNeighbors=5)

    # Use multiprocessing to parallelize the face detection process
    pool = multiprocessing.Pool(processes=3)
    results = pool.starmap(calculate_histogram_similarity, [(face_box, gray_input_image) for face_box in detected_faces])
    pool.close()
    pool.join()

    # Output the name of the training image with the best match
    best_match_name = max(results, key=lambda x: results.count(x), default=None)

    if best_match_name:
        print("The input face matches best with the training image:", best_match_name)
        desktop_path = "C:/Users/Rameesha/Desktop"
        output_file_path = os.path.join(desktop_path, "output.txt")
        with open(output_file_path, 'w') as f:
            f.write("The input face matches best with the training image: " + best_match_name)


    else:
        print("No match found in the training images.")

# Calculate and print the execution time
    end_time = time.time()
    execution_time = end_time - start_time
    print("parallel Execution time:", execution_time, "ms")

if __name__ == '__main__':
    main()


print("Current process ID:", os.getpid())