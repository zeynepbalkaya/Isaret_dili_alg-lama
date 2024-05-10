import cv2
import mediapipe as mp
import numpy as np
import pickle

# Eğitilmiş modelin yüklenmesi
model_dict = pickle.load(open("model.p", "rb"))
model = model_dict["model"]

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

label_dict = {0: "A", 1: "B", 2: "C", 3: "D"}

while True:
    ret, frame = cap.read()
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmark in results.multi_hand_landmarks:
            data_aux = []  # Her el için yeni bir veri listesi oluşturuyoruz

            for i in range(len(hand_landmark.landmark)):
                x = hand_landmark.landmark[i].x
                y = hand_landmark.landmark[i].y

                # Tüm X ve Y koordinatlarını data_aux'a ekliyoruz
                data_aux.append(x)
                data_aux.append(y)

            # Eğitimde kullanılan özellik sayısını kontrol etme ve uyarlama
            while len(data_aux) < 42:  # Modelin eğitiminde kullanılan özellik sayısı 21'den 42'ye çıkarıldıysa
                data_aux.append(0.0)  # Yeni özellikler için 0'lar ekle

            data_aux = data_aux[:42]  # Modelin beklentisi olan özellik sayısını (42) kontrol etme

            prediction = model.predict([np.array(data_aux)])
            predicted_char = label_dict[int(prediction[0])]

            cv2.putText(frame, predicted_char, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()




           
