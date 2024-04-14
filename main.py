import cv2
import mediapipe as mp

# Define hand landmark detection parameters
mp_hands = mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Define fingertip IDs for index finger
TIP_IDS = [mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]

# Define screen size (replace with your actual screen dimensions)
screenWidth = 1920
screenHeight = 1080

def detect_finger(frame):
  """Detects the index finger tip in a frame and returns its (x, y) coordinates.

  Args:
      frame: The frame to be analyzed.

  Returns:
      A tuple containing the (x, y) coordinates of the index finger tip, 
      or None if not detected.
  """
  # Convert frame to RGB format
  rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  
  # Detect hands in the frame
  results = mp_hands.process(rgb_frame)

  # Get hand landmarks
  hand_landmarks = results.multi_hand_landmarks
  
  # Check if any hands are detected
  if hand_landmarks:
    for hand_landmarks in results.multi_hand_landmarks:
      # Get fingertip coordinates
      finger_tips = []
      for id_, lm in enumerate(hand_landmarks.landmark):
          # Extract landmarks based on ID
          if id_ in TIP_IDS:
              # Scale landmark coordinates based on screen size
              x = int(lm.x * screenWidth)
              y = int(lm.y * screenHeight)
              finger_tips.append((x, y))
      
      # Check if fingertip is detected and return its coordinates  
      if finger_tips:
          return finger_tips[0]  # Assuming only one finger is raised
  
  return None

# Function to move cursor based on fingertip coordinates
def move_cursor(fingertip):
  """Moves the PC cursor to the provided fingertip coordinates.

  Args:
      fingertip: A tuple containing the (x, y) coordinates of the fingertip.
  """
  if fingertip:
    x, y = fingertip
    # Use appropriate libraries to move cursor based on OS (e.g., pyautogui for Windows/macOS)
    # Replace this line with the appropriate cursor control library
    # pyautogui.moveTo(x, y)
    print(f"Cursor moved to: ({x}, {y})")  # Placeholder until cursor control is implemented

# Main program loop
cap = cv2.VideoCapture(0)  # Change 0 to camera ID if using external camera

while True:
  # Capture frame-by-frame
  ret, frame = cap.read()

  # Detect fingertip
  fingertip = detect_finger(frame)

  # Move cursor if fingertip is detected
  move_cursor(fingertip)
  
  # Display the resulting frame
  cv2.imshow('Finger Tracking', frame)

  # Exit on 'q' key press
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

# Release capture and close all windows
cap.release()
cv2.destroyAllWindows()

print("Program exited successfully!")
