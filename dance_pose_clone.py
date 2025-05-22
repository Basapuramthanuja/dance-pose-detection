import cv2
import mediapipe as mp
import numpy as np

# Setup MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Stylish drawing specs
pose_line = mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2)
pose_dot = mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=0, circle_radius=0)  # Hide dots

# Webcam setup
cap = cv2.VideoCapture(0)
cv2.namedWindow("Neon Pose", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Neon Pose", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        # Flip image
        frame = cv2.flip(frame, 1)

        # Make black canvas
        black = np.zeros_like(frame)

        # Convert to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        # Draw pose on black background
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                black,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=pose_dot,
                connection_drawing_spec=pose_line
            )

        # Apply glow effect (optional)
        glow = cv2.addWeighted(black, 1.5, cv2.GaussianBlur(black, (0, 0), 10), 0.5, 0)

        # Show final result
        cv2.imshow("Neon Pose", glow)

        # Exit on Esc
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()