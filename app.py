import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import mediapipe as mp

class CatDogClassifier:
    def __init__(self):
        """Initialize the classifier with pre-trained models"""
        # Load pre-trained MobileNetV2 model (trained on ImageNet)
        print("Loading MobileNetV2 model...")
        self.model = MobileNetV2(weights='imagenet', include_top=True)
        print("Model loaded successfully!")
        
        # Initialize MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5)
        
        # Define ImageNet class indices for cats and dogs
        # ImageNet has multiple cat and dog breeds, we'll check for these
        self.cat_classes = {
            'Egyptian_cat', 'tabby', 'tiger_cat', 'Persian_cat', 'Siamese_cat',
            'Maine_Coon', 'Angora', 'cougar', 'lynx', 'leopard', 'jaguar',
            'lion', 'tiger', 'cheetah'
        }
        
        self.dog_classes = {
            'Chihuahua', 'Japanese_spaniel', 'Maltese_dog', 'Pekinese',
            'Shih-Tzu', 'Blenheim_spaniel', 'papillon', 'toy_terrier',
            'Rhodesian_ridgeback', 'Afghan_hound', 'basset', 'beagle',
            'bloodhound', 'bluetick', 'black-and-tan_coonhound', 'Walker_hound',
            'English_foxhound', 'redbone', 'borzoi', 'Irish_wolfhound',
            'Italian_greyhound', 'whippet', 'Ibizan_hound', 'Norwegian_elkhound',
            'otterhound', 'Saluki', 'Scottish_deerhound', 'Weimaraner',
            'Staffordshire_bullterrier', 'American_Staffordshire_terrier',
            'Bedlington_terrier', 'Border_terrier', 'Kerry_blue_terrier',
            'Irish_terrier', 'Norfolk_terrier', 'Norwich_terrier',
            'Yorkshire_terrier', 'wire-haired_fox_terrier', 'Lakeland_terrier',
            'Sealyham_terrier', 'Airedale', 'cairn', 'Australian_terrier',
            'Dandie_Dinmont', 'Boston_bull', 'miniature_schnauzer',
            'giant_schnauzer', 'standard_schnauzer', 'Scotch_terrier',
            'Tibetan_terrier', 'silky_terrier', 'soft-coated_wheaten_terrier',
            'West_Highland_white_terrier', 'Lhasa', 'flat-coated_retriever',
            'curly-coated_retriever', 'golden_retriever', 'Labrador_retriever',
            'Chesapeake_Bay_retriever', 'German_short-haired_pointer',
            'vizsla', 'English_setter', 'Irish_setter', 'Gordon_setter',
            'Brittany_spaniel', 'clumber', 'English_springer', 'Welsh_springer_spaniel',
            'cocker_spaniel', 'Sussex_spaniel', 'Irish_water_spaniel',
            'kuvasz', 'schipperke', 'groenendael', 'malinois', 'briard',
            'kelpie', 'komondor', 'Old_English_sheepdog', 'Shetland_sheepdog',
            'collie', 'Border_collie', 'Bouvier_des_Flandres', 'Rottweiler',
            'German_shepherd', 'Doberman', 'miniature_pinscher',
            'Greater_Swiss_Mountain_dog', 'Bernese_mountain_dog',
            'Appenzeller', 'EntleBucher', 'boxer', 'bull_mastiff',
            'Tibetan_mastiff', 'French_bulldog', 'Great_Dane',
            'Saint_Bernard', 'Eskimo_dog', 'malamute', 'Siberian_husky',
            'affenpinscher', 'basenji', 'pug', 'Leonberg', 'Newfoundland',
            'Great_Pyrenees', 'Samoyed', 'Pomeranian', 'chow',
            'keeshond', 'Brabancon_griffon', 'Pembroke', 'Cardigan',
            'toy_poodle', 'miniature_poodle', 'standard_poodle',
            'Mexican_hairless', 'dingo'
        }
    
    def preprocess_image(self, image):
        """Preprocess image for MobileNetV2"""
        # Resize image to 224x224 (MobileNetV2 input size)
        resized = cv2.resize(image, (224, 224))
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        # Add batch dimension
        input_image = np.expand_dims(rgb_image, axis=0)
        # Preprocess for MobileNetV2
        preprocessed = preprocess_input(input_image.astype(np.float32))
        return preprocessed
    
    def classify_image(self, image):
        """Classify image as Cat, Dog, or Not Cat/Dog"""
        # Preprocess the image
        preprocessed_image = self.preprocess_image(image)
        
        # Make prediction
        predictions = self.model.predict(preprocessed_image, verbose=0)
        
        # Decode predictions to get class names
        decoded_predictions = decode_predictions(predictions, top=5)[0]
        
        # Check if any of the top predictions are cats or dogs
        for _, class_name, confidence in decoded_predictions:
            if confidence > 0.1:  # Only consider predictions with >10% confidence
                # Clean class name (remove underscores, convert to lowercase)
                clean_class_name = class_name.replace('_', ' ').lower()
                
                # Check if it's a cat
                if any(cat_class.lower().replace('_', ' ') in clean_class_name or 
                       clean_class_name in cat_class.lower().replace('_', ' ') 
                       for cat_class in self.cat_classes):
                    return "Cat", confidence
                
                # Check if it's a dog
                if any(dog_class.lower().replace('_', ' ') in clean_class_name or 
                       clean_class_name in dog_class.lower().replace('_', ' ') 
                       for dog_class in self.dog_classes):
                    return "Dog", confidence
        
        return "Not Cat/Dog", decoded_predictions[0][2]  # Return with highest confidence
    
    def detect_faces(self, image):
        """Detect faces using MediaPipe"""
        # Convert BGR to RGB for MediaPipe
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        results = self.face_detection.process(rgb_image)
        
        face_boxes = []
        if results.detections:
            for detection in results.detections:
                # Get bounding box
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = image.shape
                x = int(bboxC.xmin * iw)
                y = int(bboxC.ymin * ih)
                w = int(bboxC.width * iw)
                h = int(bboxC.height * ih)
                face_boxes.append((x, y, w, h))
        
        return face_boxes
    
    def run_webcam_classification(self):
        """Main function to run webcam with real-time classification"""
        # Initialize webcam
        print("Initializing webcam...")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        print("Webcam initialized successfully!")
        print("Press 'q' to quit")
        
        # Variables for performance optimization
        frame_count = 0
        classification_interval = 10  # Classify every 10 frames for better performance
        last_classification = "Not Cat/Dog"
        last_confidence = 0.0
        
        while True:
            # Read frame from webcam
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Detect faces (optional - for demonstration)
            face_boxes = self.detect_faces(frame)
            
            # Draw face detection boxes (green rectangles)
            for (x, y, w, h) in face_boxes:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, "Face Detected", (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Perform classification every N frames for better performance
            if frame_count % classification_interval == 0:
                try:
                    classification, confidence = self.classify_image(frame)
                    last_classification = classification
                    last_confidence = confidence
                except Exception as e:
                    print(f"Classification error: {e}")
                    last_classification = "Error"
                    last_confidence = 0.0
            
            # Display classification result
            result_text = f"{last_classification} ({last_confidence:.2f})"
            
            # Choose color based on classification
            if last_classification == "Cat":
                color = (0, 165, 255)  # Orange for cats
            elif last_classification == "Dog":
                color = (255, 0, 0)    # Blue for dogs
            else:
                color = (0, 0, 255)    # Red for not cat/dog
            
            # Draw classification result on frame
            cv2.putText(frame, result_text, (30, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
            
            # Add instructions
            cv2.putText(frame, "Press 'q' to quit", (30, frame.shape[0] - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display frame
            cv2.imshow('Cat/Dog Classifier', frame)
            
            # Check for quit key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_count += 1
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("Webcam closed successfully!")

def main():
    """Main function to run the application"""
    print("=== Real-time Cat/Dog Classifier ===")
    print("This application uses:")
    print("- OpenCV for webcam access")
    print("- MediaPipe for face detection")
    print("- Pre-trained MobileNetV2 for cat/dog classification")
    print("\nStarting application...")
    
    try:
        # Create classifier instance
        classifier = CatDogClassifier()
        
        # Run webcam classification
        classifier.run_webcam_classification()
        
    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have all required packages installed:")
        print("pip install opencv-python tensorflow mediapipe numpy")

if __name__ == "__main__":
    main()