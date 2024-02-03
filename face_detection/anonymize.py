import cv2

import mediapipe as mp
from blur import blur_faces


def main():
    mp_face_detection = mp.solutions.face_detection

    cap = cv2.VideoCapture(0)
    while True:
        _, frame = cap.read()

        with mp_face_detection.FaceDetection(
                min_detection_confidence=0.5,
                model_selection=0) as face_det:
            proc = blur_faces(frame, face_det)

            if proc is not None:
                frame = proc
            
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()
