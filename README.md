# 🤖 MuJoCo TurtleBot3 VLA

**Gemini 기반 자연어 명령 + YOLO 객체 인식 + LiDAR(Rangefinder) + Wheel Encoder**를 결합하여 TurtleBot3를 제어하는 MuJoCo 시뮬레이션 프로젝트입니다.

사용자의 자연어 명령을 Gemini가 로봇 제어 명령으로 변환하고, 카메라 기반 객체 인식과 센서 정보를 활용하여 **주행, 객체 탐색, 장애물 회피**를 수행합니다.

---

## 🎯 주요 기능

* **LLM 기반 자연어 로봇 제어**

  * Gemini를 이용하여 사용자의 자연어 명령을 주행 명령으로 변환
  * 직진, 후진, 좌·우회전, 제자리 회전 및 지정 각도 회전 지원
  * 연속 직진 명령 지원

* **YOLO 기반 객체 인식**

  * 로봇 카메라 영상에서 과일 카드 인식
  * Apple / Orange / Banana / Watermelon 인식
  * 특정 과일이 보이지 않을 경우 해당 객체를 찾기 위한 탐색 수행

* **LiDAR(Rangefinder) 기반 장애물 회피**

  * 전방 장애물과의 거리 측정
  * 장애물이 1 m 이내로 접근하면 자동 회피 모드로 전환
  * 좌·우 방향의 거리 비교를 통해 회피 방향 결정

* **Wheel Encoder 기반 회전 제어**

  * Wheel Encoder를 이용해 로봇의 회전량 계산
  * 목표 각도에 도달할 때까지 회전량을 피드백하여 회전
  * 단순 회전 방식과 정밀 회전 방식을 구분하여 구현

---

## 🏗️ 시스템 구조

```text
              User
                │
                │ Natural Language
                ↓
        ┌────────────────┐
        │     Gemini     │
        │  LLM Command   │
        └───────┬────────┘
                │
                ↓
┌────────────────────────────────┐
│        TurtleBot3 / MuJoCo     │
│                                │
│  ┌─────────┐    ┌───────────┐  │
│  │   YOLO  │    │ Rangefinder│ │
│  │ Camera  │    │ + Encoder │  │
│  └────┬────┘    └─────┬─────┘  │
│       │               │        │
│       └───────┬───────┘        │
│               ↓                │
│        Motion Control          │
└────────────────┬───────────────┘
                 ↓
          Robot Movement
```

---

## 🧠 LLM 기반 제어

Gemini는 사용자 입력과 현재 환경 정보를 바탕으로 다음 행동 중 하나를 선택합니다.

```text
직진
후진
좌회전 / 우회전
제자리 회전
TURN_LEFT_N / TURN_RIGHT_N
SEARCH_<OBJECT>
```

## 👀 객체 탐색

**객체 인식 결과가 필요한 경우 YOLO의 Detection 결과를 JSON 형태로 Gemini에 전달하고, 전방 Rangefinder 거리도 함께 제공하여 상황에 맞는 행동을 결정하도록 구성했습니다.**

사용자가 특정 과일을 요청했지만 해당 객체가 현재 카메라에 보이지 않는 경우, Gemini를 호출하기 전에 목표 객체에 대응하는 `SEARCH_*` 명령을 생성합니다.

```text
"바나나 카드 찾아봐"
        ↓
YOLO Detection
        ↓
Banana 미검출
        ↓
SEARCH_BANANA
        ↓
제자리 회전하며 탐색
        ↓
Banana 발견 → 정지
```

## 🚧 장애물 회피

**탐색 중 목표 객체가 YOLO에서 검출되면 회전을 멈추고 탐색을 종료합니다.**

사용자가 `계속 직진`을 명령하면 로봇은 지속적으로 전진하면서 전방 Rangefinder를 확인합니다.

전방 거리가 **1.0 m 미만**이 되면 자동으로 회피 모드로 전환됩니다.

### 회피 과정

```text
장애물 감지
    ↓
오른쪽 60° 회전
    ↓
전방 거리 측정
    ↓
왼쪽 120° 회전
    ↓
전방 거리 측정
    ↓
두 거리 비교
    ↓
더 넓은 방향으로 90° 회전
    ↓
직진 재개
```

두 방향의 거리가 비슷한 경우에는 주변 공간에 따라 90° 또는 180° 회전을 수행합니다.

---

## 🔄 Wheel Encoder 기반 회전 제어

Wheel Encoder의 좌·우 바퀴 회전량을 이용하여 로봇의 회전량을 계산하는 방식을 구현했습니다.

```text
Left Encoder ─┐
              ├→ Wheel Rotation → Robot Yaw
Right Encoder ┘
```

또한 MuJoCo의 실제 Body Orientation을 이용하여 현재 yaw를 계산하고, 목표 각도와의 오차에 따라 회전 속도를 조절하는 방식도 구현했습니다. 이를 통해 목표 각도에 가까워질수록 회전 속도를 줄이고 설정된 오차 범위 내에서 회전을 종료합니다.

---

## 📁 프로젝트 구조

```text
vla_lidar-wheelencoder/
│
├── asset/
│   └── robotis_tb3/
│       └── tb3_factory_cards.xml
│
├── scripts/
│   ├── tb3_vla.py
│   ├── tb3_sim.py
│   ├── gemini_tb3.py
│   └── prompt.yaml
│
├── utils/
│   ├── mujoco_renderer.py
│   └── object_detector.py
│
├── README.md
└── LICENSE
```

---

## ▶️ 실행 방법

필요한 Python 패키지와 Gemini API Key를 설정한 후 `scripts/tb3_vla.py`를 실행합니다.

```bash
python scripts/tb3_vla.py
```

`tb3_vla.py`에서 MuJoCo 환경, YOLO 모델, Gemini Agent를 초기화한 뒤 LLM 스레드와 시뮬레이션 루프를 실행합니다.

---
