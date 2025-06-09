from model import get_model
import torch
import utils
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image

if __name__ == '__main__':
    config = utils.read_config()
    save_path = config['model_save_path']
    model = get_model('ResNet18', num_classes=7)

    # 加载 checkpoint 并提取模型权重
    checkpoint = torch.load(save_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    model = model.to(config['device'])

    img_size = config['img_size']
    # 更新数据转换，添加标准化处理
    data_trans = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    label_name = ['Surprise', 'Fear', 'Disgust', 'Happiness', 'Sadness', 'Anger', 'Neutral']
    device = config['device']
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    model.eval()
    
    while True:
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces_rects = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            for (x, y, w, h) in faces_rects:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # 直接使用彩色图像进行处理
                face_img = frame[y:y+h, x:x+w]
                # 转换为RGB格式
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                face_img = Image.fromarray(face_img)
                face_img = data_trans(face_img)
                face_img = face_img.unsqueeze(0)
                face_img = face_img.to(device)
                
                with torch.no_grad():
                    label_pd = model(face_img)
                    # 获取预测概率
                    probabilities = torch.nn.functional.softmax(label_pd, dim=1)
                    # 获取最大概率和对应的类别
                    max_prob, predict_np = torch.max(probabilities, 1)
                    confidence = max_prob.item()
                    
                    # 只有当置信度大于阈值时才显示预测结果
                    if confidence > 0.5:  # 可以调整这个阈值
                        fer_text = f"{label_name[predict_np[0]]} ({confidence:.2f})"
                    else:
                        fer_text = "Unknown"
                
                font = cv2.FONT_HERSHEY_SIMPLEX
                pos = (x, y - 10)  # 将文字位置稍微上移
                font_size = 0.8
                color = (0, 0, 255)
                thickness = 2
                cv2.putText(frame, fer_text, pos, font, font_size, color, thickness,
                           cv2.LINE_AA)
            
            cv2.imshow('Face Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    
    cap.release()
    cv2.destroyAllWindows()