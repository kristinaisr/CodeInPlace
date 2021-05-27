import cv2

def face_detection(imageName):
    sourceFileName = "source_images/"
    finalFileName = "final_images/"
    
    # Load the cascade
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_1.xml')
    
    # Read the input image
    img = cv2.imread(sourceFileName + imageName)
    
    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    # Draw rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
    # Display the output
    cv2.imwrite(finalFileName+'Face_'+imageName,img)
    cv2.imshow('img', img)
    cv2.waitKey()
    
    


def main():
    imageNamesList = ["photo-1.jpeg", "photo-2.jpeg", "photo-3.jpeg"]
    for imageName in imageNamesList:
        face_detection(imageName)
        
        
if __name__ == '__main__':
    main()