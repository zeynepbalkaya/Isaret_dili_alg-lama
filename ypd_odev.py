#İŞARET DİLİ ALFABESİ ALGILAMA PROJESİ 
#ZEYNEP BALKAYA 202013171010
#RUKİYE BEYZA TÜRKEN 202013171036
#DOĞAN ESEN 201913172030 (İkinci Öğretim)
import os
import cv2

DATA_DIR = "./data"

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_class = 4
data_size = 100

cap = cv2.VideoCapture(0)

for j in range(number_of_class):
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print("Sinif {}".format(j))
    print("Hazir olunca q basin.")

    while True:
        _, frame = cap.read()
        cv2.putText(frame, "Hazir olunca q basin", (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3)
        cv2.imshow("frame", frame)

        if cv2.waitKey(25) & 0xFF == ord("q"):
            break

    print("Görüntüler aliniyor...")
    counter = 0
    while counter < data_size:
        ret, frame = cap.read()
        cv2.imshow("frame", frame)
        cv2.waitKey(25)
        img_path = os.path.join(class_dir, "{}.jpg".format(counter))
        cv2.imwrite(img_path, frame)
        counter += 1

cap.release()
cv2.destroyAllWindows()