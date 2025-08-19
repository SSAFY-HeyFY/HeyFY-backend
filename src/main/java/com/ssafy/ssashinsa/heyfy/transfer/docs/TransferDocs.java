package com.ssafy.ssashinsa.heyfy.transfer.docs; // 기존 docs 패키지에 생성

import com.ssafy.ssashinsa.heyfy.common.exception.ErrorResponse;
import com.ssafy.ssashinsa.heyfy.transfer.dto.TransferHistoryResponse;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.media.Content;
import io.swagger.v3.oas.annotations.media.ExampleObject;
import io.swagger.v3.oas.annotations.media.Schema;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import io.swagger.v3.oas.annotations.responses.ApiResponses;

import java.lang.annotation.Documented;
import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

@Target(ElementType.METHOD)
@Retention(RetentionPolicy.RUNTIME)
@Documented
@Operation(summary = "계좌 이체", description = "출금 계좌에서 입금 계좌로 금액을 이체합니다.")
@ApiResponses({
        // 🔹 1. 성공 응답 (200 OK)
        @ApiResponse(
                responseCode = "200",
                description = "이체 성공",
                content = @Content(
                        mediaType = "application/json",
                        schema = @Schema(implementation = TransferHistoryResponse.class) // 실제 반환 타입 명시
                )
        ),
        // 🔹 2. 실패 응답 (400 Bad Request)
        @ApiResponse(
                responseCode = "400",
                description = "잘못된 요청",
                content = @Content(
                        mediaType = "application/json",
                        schema = @Schema(implementation = ErrorResponse.class),
                        examples = {
                                // OpenApiConfig에 정의된 공통 예시 참조
                                @ExampleObject(
                                        name = "계좌번호 누락",
                                        ref = "#/components/examples/MissingRequired"
                                ),
                                // 이 API에서만 발생하는 특정 에러 예시를 직접 작성
                                @ExampleObject(
                                        name = "잘못된 이체 금액",
                                        summary = "금액이 0 이하인 경우",
                                        value = "{\"code\": 400, \"message\": \"이체 금액은 0보다 커야 합니다.\"}"
                                ),
                                @ExampleObject(
                                        name = "외부 API 호출 실패",
                                        summary = "은행사 응답 에러",
                                        value = "{\"code\": 400, \"message\": \"[H0001] 잔액이 부족합니다.\"}"
                                )
                        }
                )
        )
})
public @interface TransferDocs {
}