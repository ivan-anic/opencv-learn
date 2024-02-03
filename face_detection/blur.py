import cv2

    
def blur_faces(img, face_det):
    """
    Blurs all of the detected faces in an image
    
    Args:
        age1 (ndarray): image
        age2 (mp.facedet): mediapipe face processing object
    
    Returns:
        img (ndarray): the processed image
    """
    H, W, _ = img.shape
    
    img_rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_det.process(img_rgb)
    if results.detections is not None:          
        for detection in results.detections:
            location_data = detection.location_data
            bbox = location_data.relative_bounding_box
            x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height
            x1 = max(int(x1*W), 0)
            y1 = max(int(y1*H), 0)
            w = int(w*W)    
            h = int(h*H)

            # blur faces
            img[y1:y1+h, x1:x1+w, :] = cv2.blur(img[y1:y1+h, x1:x1+w, :], (40, 40))

        return img
    