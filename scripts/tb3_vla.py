import os
from queue import Queue

from scripts.tb3_sim import TurtlebotFactorySim
from scripts.gemini_tb3 import GeminiTb3

PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), ".."))
xml_path = "/home/thj326/vla_project/asset/robotis_tb3/tb3_factory_cards.xml"
prompt_path = "/home/thj326/vla_project/scripts/prompt.yaml"
yolo_weights = "/home/thj326/vla_project/scripts/best.pt"

cmd_q = Queue()

# 1) 터틀봇 + YOLO 시뮬
sim = TurtlebotFactorySim(
    xml_path=xml_path,
    use_yolo=True,
    yolo_weight_path=yolo_weights,
    yolo_conf=0.4,
    command_queue=cmd_q,
    fps=60,
)

# 2) Gemini + YOLO + 명령 생성
agent = GeminiTb3(
    prompt_path=prompt_path,
    model="gemini-robotics-er-1.5-preview",
    command_queue=cmd_q,
)

# 3) LLM 쓰레드 시작
agent.start(sim)

# 4) 시뮬 루프 시작 (키보드로 q 누르면 종료)
sim.start()
