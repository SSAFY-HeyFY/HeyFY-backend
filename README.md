# HeyFY-backend
신한은행 해커톤 3기 백엔드 레포입니다.

# 📄 Swagger 커스텀 응답 작성 가이드

Swagger(OpenAPI) 문서에서 **공통 오류 응답**과 **API별 커스텀 응답**을 분리해 관리합니다.  
클래스 레벨에서는 인증/인가 관련 공통 오류만, 메서드 레벨에서는 성공 및 400 오류 응답을 작성합니다.

---

## 🛠 작성 규칙

### 1. 클래스 상단에 `@ErrorsCommon` 등록
- 모든 컨트롤러 클래스 상단에 **공통 오류 응답 애너테이션** `@ErrorsCommon`을 붙입니다.
- `ErrorsCommon`에는 **401(Unauthorized)**, **403(Forbidden)** 응답만 정의합니다.
- **400 응답은 선언하지 않습니다** → 메서드에서 커스텀하기 위함입니다.
---

### 2. 메서드별 **커스텀 애너테이션** 생성
- 각 API 메서드 전용으로 **커스텀 애너테이션**을 만듭니다.
   - 애너테이션 명 : {controller명}+{method명}+Docs
   - ex) AuthSignInDocs
- 포함 내용:
    - **성공 응답** (`200`, `201` 등) – 응답 스키마 및 예시
    - **실패 응답 (400)** – 요청 유효성 오류나 비즈니스 로직 위반 등의 예시
<details>
<summary>커스텀 애너테이션 예시</summary>
<div markdown="1">

```java
@Target(ElementType.METHOD)
@Retention(RetentionPolicy.RUNTIME)
@Documented
@ApiResponses({
    // 🔹 성공 응답(메서드별 정의)
    @ApiResponse(
        responseCode = "200",
        description = "메뉴 수정 성공",
        content = @Content(
            mediaType = "application/json",
            schema = @Schema(implementation = ResultSuccessResponseDto.class),
            examples = {
                @ExampleObject(
                    name = "SuccessExample",
                    value = "{\"success\":true,\"message\":\"OK\"}"
                )
            }
        )
    ),
    // 🔹 실패 400 응답(메서드별 커스텀)
    @ApiResponse(
        responseCode = "400",
        description = "잘못된 요청",
        content = @Content(
            mediaType = "application/json",
            schema = @Schema(implementation = ErrorResponse.class),
            examples = {
                // 공통 예시 재사용 (ref만 단독 사용)
                @ExampleObject(ref = "#/components/examples/MissingRequired"),
                @ExampleObject(ref = "#/components/examples/NotParticipant"),
                // 필요 시 인라인 예시 추가
                @ExampleObject(
                    name = "InvalidPrice",
                    value = "{\"code\":4003,\"message\":\"price는 0 이상이어야 합니다.\"}"
                )
            }
        )
    )
})
public @interface ExampleDocs {}
```
</div>
</details>

---

### 3. 메서드에 커스텀 애너테이션 등록
- 컨트롤러 메서드에 해당 API용 커스텀 애너테이션을 붙입니다.
- 클래스에 선언된 401/403 공통 응답과 병합되어 문서에 표시됩니다.
<details>
<summary>컨트롤러 적용 예시</summary>
<div markdown="1">

```java
@ErrorsCommon       // ← 공통 응답(401,403) 등록
@RestController
@RequestMapping(value = "/api/example", produces = MediaType.APPLICATION_JSON_VALUE)
public class ExampleApiController {

    @PatchMapping("/{id}")
    @ExampleDocs    // ← 메서드 전용 메타 애너테이션 등록
    public ResultSuccessResponseDto update(
            @PathVariable Long id,
            @RequestBody RequestDto request
    ) {
        return ResultSuccessResponseDto.ok();
    }
}
```
</div>
</details>

---

## 💡 운영 팁
- **예시 재사용**: 반복적으로 쓰이는 예시는 `components.examples`에 등록하고 `ref`로 참조
- **충돌 방지**: 같은 상태코드를 클래스·메서드 양쪽에 선언하지 않음
- **미디어 타입 통일**: `mediaType = "application/json"`로 일관성 유지
- **스키마 명시**: `schema = @Schema(implementation = ...)`로 응답 구조를 명확히 지정
- **규칙 준수**: 모든 API에서 동일한 패턴 적용

# 🤫 application.yml 관리
- `application.yml` 파일 수정 금지
- `application-local.yml` 수정 시 리뷰 후 반영
  - ❗️appKey는 공백❗️(각자 관리)
- `application-secret.yml`은 @EomYoosang 만 관리

# 🐋 Docke-Compose를 사용한 로컬 환경 설정
1. Docker 설치
   - [도커 사이트](https://www.docker.com/) 접속하여 OS에 맞는 도커 설치
2. Docker hub 로그인
3. 프로젝트 루트 디렉토리에서 명령어 실행
```
docker-compose -f docker-compose-local.yml up -d
```

# 🌿 Git Branch 전략 및 커밋 컨벤션

## 📌 브랜치 종류 및 규칙

| 브랜치 | 용도 | 설명 |
|--------|------|------|
| `main` | 배포용 | 항상 **안정적인 상태** 유지. 배포 시 이 브랜치 기준으로 진행. **직접 작업 금지.** |
| `develop` | 개발 통합 | 각 기능 브랜치를 이 브랜치로 merge. 팀원 PR 후 코드리뷰 → merge 권장. 리뷰 지연 시 **자기 책임 하에 직접 merge 가능.** |
| `feature/{이슈번호}-{설명}` | 기능 개발 | 새로운 기능 개발 시 사용. 예: `feature/#12-login-api` |
| `fix/{이슈번호}-{설명}` | 버그 수정 | 발견된 버그 수정용 브랜치. |
| `hotfix/{이슈번호}-{설명}` | 긴급 수정 | 배포 후 발생한 긴급 이슈 처리 시 사용. |
| `refactor/{이슈번호}-{설명}` | 리팩토링 | 로직 변경 없이 코드 구조 개선 목적. |
| `chore/{이슈번호}-{설명}` | 설정/환경 | 빌드 설정, 패키지 설치 등 부수 작업 시 사용. |

> ✅ 기능 개발이 완료되면 `develop` 브랜치로 **Pull Request → Merge**  
> ✅ merge 완료된 브랜치는 **즉시 삭제 권장**

---

## ✅ 브랜치 네이밍 예시

- `feature/#12-login-api`
- `fix/#17-cors-error`
- `chore/#20-env-setting`

---

## 📝 커밋 메시지 컨벤션

### 🔧 커밋 메시지 형식

```
#{ISSUE_NUMBER}: 변경 요약
```

* 이슈가 없는 작업은 [NO-ISSUE]

> 예시:  
> `#10: 로그인 API 구현`

---

### 📚 ISSUE 생성 방식

-  큰 Task 로 부모 ISSUE 생성
-  부모 ISSUE 에 맞는 Sub Task 생성 (최대한 작게 나누기)


## 🔁 브랜치 Workflow 요약

1. 이슈 생성  
2. `develop` 브랜치 기준으로 기능 브랜치 생성  
   ```bash
   git checkout -b feature/#12-login-api develop
   ```

3. 기능 개발 후 커밋 & 푸시  
   ```bash
   git commit -m "#12 [feat]: 로그인 API 구현"
   git push origin feature/#12-login-api
   ```

4. GitHub에서 Pull Request 생성  
5. 코드리뷰 후 merge (지연 시 셀프 merge 가능)  
6. merge된 브랜치는 삭제  

---
