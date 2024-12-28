import cv2
import mediapipe as mp
import time

# Initialize MediaPipe Holistic
mpHolistic = mp.solutions.holistic
mpHands = mp.solutions.hands
holistic = mpHolistic.Holistic()
mpDraw = mp.solutions.drawing_utils

# Drawing specifications
poseSpec = mpDraw.DrawingSpec(color=(0, 0, 204), thickness=2, circle_radius=2)
handSpec = mpDraw.DrawingSpec(color=(255, 153, 204), thickness=2, circle_radius=2)
faceSpec = mpDraw.DrawingSpec(color=(102, 178, 255), thickness=1, circle_radius=1)

# Finger tip landmarks (MediaPipe indexes)
FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_MCP = [2, 5, 9, 13, 17]  # MCP landmarks for each finger

# Open the video capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot open camera.")
    exit()

# Initialize time variables for FPS calculation
prevTime = 0
currTime = 0

while True:
    success, img = cap.read()
    if not success:
        print("Failed to grab frame.")
        break

    # Convert image to RGB for MediaPipe
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process the frame with Holistic
    results = holistic.process(imgRGB)

    # Initialize finger count
    fingerCount = 0

    # Process hand landmarks for finger counting
    if results.left_hand_landmarks or results.right_hand_landmarks:
        for hand_landmarks in [results.left_hand_landmarks, results.right_hand_landmarks]:
            if hand_landmarks:
                landmarks = hand_landmarks.landmark

                # Thumb logic (use x-coordinate for left/right check)
                if landmarks[FINGER_TIPS[0]].x < landmarks[FINGER_MCP[0]].x:
                    fingerCount += 1

                # Other fingers (use y-coordinate for tip vs PIP position)
                for i in range(1, 5):  # Index to Pinky
                    if landmarks[FINGER_TIPS[i]].y < landmarks[FINGER_TIPS[i] - 2].y:
                        fingerCount += 1

    # Draw pose landmarks
    if results.pose_landmarks:
        mpDraw.draw_landmarks(
            img, results.pose_landmarks, mpHolistic.POSE_CONNECTIONS, poseSpec, poseSpec
        )

    # Draw hand landmarks
    if results.left_hand_landmarks:
        mpDraw.draw_landmarks(
            img, results.left_hand_landmarks, mpHands.HAND_CONNECTIONS, handSpec, handSpec
        )
    if results.right_hand_landmarks:
        mpDraw.draw_landmarks(
            img, results.right_hand_landmarks, mpHands.HAND_CONNECTIONS, handSpec, handSpec
        )

    # Draw face mesh
    if results.face_landmarks:
        mpDraw.draw_landmarks(
            img, results.face_landmarks, mpHolistic.FACEMESH_TESSELATION, faceSpec, faceSpec
        )

    # Calculate FPS
    currTime = time.time()
    fps = 1 / (currTime - prevTime)
    prevTime = currTime

    # Display FPS and Finger Count on the frame
    cv2.putText(
        img, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2
    )
    cv2.putText(
        img,
        f"Fingers: {fingerCount}",
        (10, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 255),
        2,
    )

    # Display the video
    cv2.imshow("Full Body Mesh with FPS and Finger Count", img)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()