# HeyFY Backend

해커톤 출품작 HeyFY 서비스의 백엔드 레포지토리입니다.

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
-   **Framework**: Spring Boot 3.2.4
-   **Build Tool**: Gradle 8.7
-   **Containerization**: Docker, Docker Compose
-   **Database**: Redis (캐싱 및 인증 토큰 관리)

## ⚙️ 빌드 및 실행 방법

### 1. Docker 사용 (권장)

프로젝트 실행에 필요한 모든 환경이 Docker Compose에 설정되어 있어 가장 간편하게 실행할 수 있습니다.

**사전 요구사항**:
-   Docker Desktop 설치

**실행 명령어**:

프로젝트 루트 디렉토리에서 다음 명령어를 실행합니다.

```bash
docker-compose -f docker-compose-local.yml up --build
```

위 명령어를 통해 프로젝트 빌드 및 Docker 이미지 생성, 그리고 컨테이너 실행이 한번에 진행됩니다.
서버는 `http://localhost:8080` 에서 실행됩니다.

### 2. 로컬에서 직접 실행

**사전 요구사항**:
-   Java 17 (JDK 17)
-   Gradle 8.7

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
    │       └── logback-spring.xml      # 로그 설정
    └── test
        └── java
            └── com/ssafy/ssashinsa/heyfy # 테스트
```