import cv2
faceCascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
video_capture=cv2.VideoCapture(0)
while True:
    ret,image=video_capture.read()
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    faces= faceCascade.detectMultiScale(gray,scaleFactor=1.1, minNeighbors=8,minSize=(30,30))
    print(f"found {len(faces)} faces.")
    for(x,y,w,h) in faces:
        cv2.rectangle(image,(x,y),(x+w, y+h),(0,255,0),3)
    cv2.imshow("Faces found", image)
    key= cv2.waitKey(1)

    if key==ord('c'):
        break
video_capture.release()
cv2.destroyAllWindows()

