# MEC and Cloud offloading optimize with Reinforcement Learning

## 환경 설정 (Setup)
이 프로젝트는 Python 가상환경(`venv`)을 사용합니다.  
아래 절차를 따라 환경을 설정하세요.

### **가상환경 생성**
먼저, Python 가상환경을 생성합니다.

#### **Windows**
```sh
python -m venv venv
venv\Scripts\activate
```

#### **Mac/Linux**
```sh
python3 -m venv venv
source venv/bin/activate
```

---

### ** 패키지 설치 (`requirements.txt` 사용)**
가상환경이 활성화된 상태에서 프로젝트의 필수 패키지를 설치합니다.
```sh
pip install -r requirements.txt
```

---

### **3️⃣ 실행 방법**
설정이 완료되면 다음 명령어로 스크립트를 실행할 수 있습니다.
```sh
python my_script.py
```

---

### **4️⃣ 가상환경 비활성화**
작업이 끝난 후 가상환경을 종료하려면:
```sh
deactivate
```

---

## 📌 **주의사항**
- Python 3.8 ~ 3.11가 설치되어 있어야 합니다.  -25년 3월 기준
  **버전 확인:**  
  ```sh
  python --version
  ```
- `requirements.txt`에 있는 패키지 목록은 `pip freeze > requirements.txt` 명령어로 업데이트할 수 있습니다.

