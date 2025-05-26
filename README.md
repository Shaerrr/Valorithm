# Valorithm
## MCP 기반 AI Agent & AI Tools를 활용한 언리얼 융합 개발 프로젝트
### 역할 구성
- **AI Engineers** : 김형섭, 이성복, 김형후
- **Unreal Engine Developers** : 이충헌, 최연택, 김병대, 김희연

### 프로젝트 기간
- 2025.04 ~ 진행 중
- <a href="https://www.notion.so/PotenUP-Project-5_Valorithm-1cf33c1194e8809a9512c785febf41f8?pvs=4">협업 기록(Notion)</a>

---

## 1. 프로젝트 개요
**Valorithm**은 AI 엔지니어 3명과 언리얼 엔진 개발자 4명이 협업하여 제작 중인 차세대 FPS 게임 개발 프로젝트입니다.  
본 프로젝트는 MCP 기반의 AI Agent 및 AI Tool을 직접 설계·구현함으로써, 언리얼 엔진 에디터 환경에서의 게임 개발 워크플로우를 혁신하고, 플레이어에게 몰입감 높은 인게임 AI 경험을 제공하는 것을 목표로 합니다.
AI 기술이 프로젝트 기획부터 개발, 플레이 환경 전반에 걸쳐 유기적으로 통합되도록 설계되었으며 개발 효율성과 사용자 경험을 동시에 향상시키는 새로운 게임 제작 패러다임을 제시합니다.

- **AI 기능.1** : 프로젝트 기획 과정에서, **회의록 정리와 일정 리마인더를 도와주는 Discord MCP기반의 AI Agent**
- **AI 기능.2** : 프로젝트 개발 과정에서, 무기의 총기 궤적을 **자연어 명령을 통해 생성하고 시각화해주는 MCP 기반의 AI Tool**
- **AI 기능.3** : 프로젝트 개발 과정에서, **2D 이미지 한장을 3D Mesh 맵으로 변경 생성**해주는 Web Server 기반의 AI Tool
- **AI 기능.4** : 프로젝트 사용 과정에서, 유저를 위한 게임 설명, 규칙, 특징의 이해를 돕는 **LangGranph 기반의 AI Agent "Javis"**

- <a href="https://www.canva.com/design/DAGoJUcpX6I/U_m7ITH1VmmHDcPLT7uVIg/view?utm_content=DAGoJUcpX6I&utm_campaign=designshare&utm_medium=link2&utm_source=uniquelinks&utlId=hf2c64f3906">프로젝트 프레젠테이션 자료</a>

---

## 2. 주요 AI 기능

### 🌀 무기 반동 궤적 생성 (Weapon Recoil Generation)  
- 총기별 설정 반동 궤적 데이터를 생성  
- Numpy 기반으로 난수를 생성하여, x/y 좌표 시퀀스 출력 및 시각화  
- Unreal MCP Plugin을 통해, 언리얼엔진5에 MCP 연동하여 자연어로 출력 가능 

🔗 <a href="https://github.com/Hyeongseob91/mcp-server.git">관련 MCP 서버 리포지토리</a>

---

### 🧍‍♂️ 실시간 캐릭터 포즈 분석 및 피드백  
- YOLO-Pose / HRNet / OpenPose를 이용하여 관절 추출 및 자세 인식  
- MCP를 통해 추출된 Pose 데이터를 LLM으로 전송 → 문제 자세 추출 및 교정 피드백  
- AI가 실시간으로 잘못된 자세를 인식하고 교정 유도  

---

### 🧠 AI 기반 적군 행동 트리 설계  
- 강화학습 + 휴리스틱 방식 결합을 통해 적군 유닛의 AI 결정 구현  
- 행동 트리는 FastAPI를 통해 독립 실행되며, 언리얼 엔진에서 REST API 방식으로 호출  

---

### 🛠️ 절차적 콘텐츠 생성 (Procedural Content Generation)  
- LLM + RAG 기반 시나리오 및 NPC 대사 자동 생성  
- LangChain 파이프라인 및 ChromaDB 벡터 저장소 활용  

---

## 3. 기술 스택

### ⚙️ Frameworks & Libraries
- LangGranph, LLM(Claude), PyTorch, OpenCV, ChromaDB, FastAPI, Streamlit, Matplotlib

### 🎮 Unreal Engine
- Unreal Engine 5 (Blueprint + C++)

### 🗂️ Database
- SQLite (로컬 저장형 RAG)

### 🚀 배포 및 협업
- Docker, Smithery MCP MarketPalce

---

## 4. 게임 개발자 & 게임 사용자 활용 시나리오

### 🎮 게임 개발자 시나리오
- AI 툴킷으로 강화학습 기반 반동 패턴 및 NPC 행동 트리 삽입
- Pose 기반 모션 캡처 데이터와 캐릭터 애니메이션 교정에 활용
- RAG 기반 퀘스트 작성 자동화로 콘텐츠 설계 시간 단축

### 👤 게임 사용자 시나리오
- 보다 사실적이고 다양하게 변화하는 적군의 전투 패턴 경험
- 플레이어 동작에 따라 피드백을 주는 훈련 시스템 경험
- 사용자 행동에 맞춘 반응형 시나리오 및 상호작용 NPC 체험

---

## 5. 설치 및 실행 방법

1. 레포지토리 클론  
   ```bash
   git clone https://github.com/chungheonLee0325/VALORANT.git
   cd VALORANT