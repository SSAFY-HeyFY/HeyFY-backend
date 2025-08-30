# HeyFY Backend

**싸신사**팀의 해커톤 출품작 HeyFY 서비스의 백엔드 레포지토리입니다.

## 🚀 프로젝트 소개

HeyFY는 환율 정보 제공 및 예측, 그리고 간편 환전 및 송금 서비스를 제공하는 핀테크 솔루션입니다.
본 레포지토리는 서비스의 핵심 비즈니스 로직을 처리하는 백엔드 API 서버입니다.

## ✨ 주요 기능

-   **사용자 인증**: JWT 기반의 안전한 사용자 회원가입 및 로그인 기능을 제공합니다.
-   **계좌 관리**: 사용자의 원화 및 외화 계좌를 등록하고 관리합니다.
-   **환율 정보**: 신한은행 API와 연동하여 실시간 환율 정보를 조회하고, 과거 환율 데이터를 제공합니다.
-   **환율 예측**: FastAPI로 구현된 AI 모델과 통신하여 미래 환율 변동을 예측합니다.
-   **환전 및 송금**: 사용자의 계좌 간 환전 및 해외 송금 기능을 제공합니다.

## 🛠️ 개발 환경

-   **Language**: Java 17
-   **Framework**: Spring Boot 3.5.4
-   **Build Tool**: Gradle 8.8
-   **Containerization**: Docker, Docker Compose
-   **Database**: Redis (캐싱 및 인증 토큰 관리)

<a id="backend-build"></a>

## ⚙️ 빌드 및 실행 방법

### 1. Docker 사용 - MySQL, Redis 실행

프로젝트 실행에 필요한 모든 환경이 Docker Compose에 설정되어 있어 가장 간편하게 실행할 수 있습니다.

**사전 요구사항**:
-   Docker Desktop 설치

**실행 명령어**:

프로젝트 루트 디렉토리에서 다음 명령어를 실행합니다.

```bash
docker-compose -f docker-compose-local.yml up --build
```

위 명령어를 통해 mysql, redis를 시킬 수 있습니다.


### 2. 로컬에서 직접 실행

**사전 요구사항**:
-   Java 17 (JDK 17)
-   Gradle 8.8
**환경변수 **
- [싸신사](https://drive.google.com/drive/folders/1lwBdA0eKgV6-9OnhEwMciOb5puspOnl4?usp=sharing)
- 위 링크를 통해 다운받은 4개의 파일을 `src/main/resources` 경로에 저장합니다.

**빌드 명령어**:

프로젝트 루트 디렉토리에서 다음 명령어를 실행하여 프로젝트를 빌드합니다.

-   **Linux/macOS**:
    ```bash
    ./gradlew build
    ```
-   **Windows**:
    ```bash
    .\gradlew.bat build
    ```

**실행 명령어**:

빌드가 완료되면 `build/libs` 디렉토리에 생성된 `.jar` 파일을 다음 명령어로 실행합니다.

```bash
java -jar build/libs/heyfy-0.0.1-SNAPSHOT.jar
```

서버는 `http://localhost:8080` 에서 실행됩니다.

## 📖 API 문서

서버가 실행되면, 아래 주소에서 API 명세(Swagger UI)를 확인할 수 있습니다.

-   [http://localhost:8080/swagger-ui/index.html](http://localhost:8080/swagger-ui/index.html)

## 📁 디렉토리 구조

프로젝트의 주요 디렉토리 구조는 다음과 같습니다.

```
.
├── build.gradle              # 프로젝트 의존성 및 빌드 설정
├── Dockerfile                # 백엔드 서버 Docker 이미지 설정
├── docker-compose-local.yml  # Docker Compose를 이용한 로컬 환경 설정
├── gradlew                   # Gradle Wrapper
└── src                       # 소스 코드
    ├── main
    │   ├── java
    │   │   └── com/ssafy/ssashinsa/heyfy
    │   │       ├── HeyfyApplication.java     # Spring Boot 메인 애플리케이션
    │   │       ├── account                   # 계좌
    │   │       ├── authentication            # 인증 및 JWT
    │   │       ├── common                    # 공통 예외 처리, 설정 등
    │   │       ├── exchange                  # 환전 및 환율 정보
    │   │       ├── fastapi                   # AI 예측 모델(FastAPI) 연동
    │   │       ├── fcm                       # FCM 푸시 알림
    │   │       ├── home                      # 홈 화면
    │   │       ├── inquire                   # 계좌/거래내역 조회
    │   │       ├── log                       # 로깅
    │   │       ├── register                  # 계좌 생성/등록
    │   │       ├── shinhanApi                # 신한은행 API 연동
    │   │       ├── swagger                   # Swagger API 문서 설정
    │   │       ├── transfer                  # 송금
    │   │       └── user                      # 사용자 정보
    │   └── resources
    │       
    └── test
        └── java
            └── com/ssafy/ssashinsa/heyfy # 테스트
```
<a id="ai-build"></a>

## AI (`ai_model/`)

AI 모델은 과거 환율 데이터를 학습하여 미래 환율을 예측하고 환전 타이밍을 추천하는 역할을 합니다. FastAPI를 통해 API 형태로 제공됩니다.

### 4.1. 개발 환경

- **Language:** `Python 3.10.4`
- **Framework:** `FastAPI`
- **의존성 관리:** `pip` 및 `requirements.txt`

### 4.2. 모델 다운로드

[AI 모델 다운로드 링크](https://drive.google.com/drive/folders/1gfALOI0HOB9SnkZpQcIFCVsNHhhImDVD?usp=sharing)

파일을 다운받고 다음 경로에 저장하면 됩니다.

- `ai_model/models/prophet_1day_model`: 1일 예측 Prophet 모델
- `ai_model/models/prophet_multi_day_model`: 2일 이상 예측 Prophet 모델

### 4.3. 빌드 및 실행 (AI 서버 단독)

0. **ai 브랜치 클론**

    ```bash
    git clone -b ai https://github.com/SSAFY-HeyFY/HeyFY-backend.git
    ```

1.  **디렉토리 이동**

    ```bash
    cd ai_model
    ```

2.  **가상 환경 생성 및 활성화 (권장)**

    ```bash
    python -m venv venv

    # macOS/Linux
    source venv/bin/activate
    
    # Windows
    venv\Scripts\activate
    ```

3.  **의존성 설치**

    `ai_model` 디렉토리에서 아래 명령어를 실행합니다.
    ```bash
    pip install -r requirements.txt
    ```

4. **📈 실시간 환율 크롤러 실행**

    ```run_realtime_crawler.py```는 현재 시점의 환율 정보를 실시간으로 크롤링하여 별도의 캐시 파일로 저장하는 스크립트입니다. 앱에서 보여주는 '현재 환율' 정보를 최신으로 유지하는 역할을 합니다.

    ```
    python scripts/run_realtime_crawler.py
    ```

5. **🤖 AI 예측 스케줄러 실행**

    ```run_ai_predictor.py```는 AI 환율 예측을 주기적으로 실행하고 결과를 캐시 파일로 저장하는 핵심 스케줄러입니다. 10분마다 다음 작업을 자동으로 수행합니다.
    학습된 하이브리드 Prophet 모델을 호출하여 미래 5일(영업일 기준)의 환율을 예측합니다. 과거 30일치 데이터와 미래 예측 데이터를 결합하여 API가 사용할 수 있도록 prediction_cache.json 파일에 저장합니다.

    ```
    python scripts/run_ai_predictor.py
    ```

6.  **AI 서버 실행**

    `ai_model` 디렉토리에서 아래 명령어를 실행하여 FastAPI 서버를 시작합니다.
    ```bash
    uvicorn app.main:app --host 0.0.0.0 --port 8888 --reload
    ```
    - `--reload` 옵션은 코드 변경 시 서버를 자동으로 재시작해 주어 개발에 유용합니다.

7.  **API 문서 확인**
    서버 실행 후, 웹 브라우저에서 `http://localhost:8888/docs` 로 접속하면 API 명세를 확인하고 직접 테스트해볼 수 있습니다.
