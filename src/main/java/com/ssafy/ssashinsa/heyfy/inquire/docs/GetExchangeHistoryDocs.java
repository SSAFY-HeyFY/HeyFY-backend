package com.ssafy.ssashinsa.heyfy.inquire.docs;

import com.ssafy.ssashinsa.heyfy.common.exception.ErrorResponse;
import com.ssafy.ssashinsa.heyfy.inquire.dto.ExchangeHistorySimplifiedDto;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.media.Content;
import io.swagger.v3.oas.annotations.media.ExampleObject;
import io.swagger.v3.oas.annotations.media.Schema;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import io.swagger.v3.oas.annotations.responses.ApiResponses;
import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

@Target(ElementType.METHOD)
@Retention(RetentionPolicy.RUNTIME)
@Operation(
        summary = "환전 내역 통합 조회",
        description = "일반 계좌와 외화 계좌의 모든 환전 내역을 통합하여 조회합니다."
)
@ApiResponses({
        @ApiResponse(
                responseCode = "200",
                description = "성공적으로 환전 내역을 통합하여 조회했습니다.",
                content = @Content(
                        mediaType = "application/json",
                        schema = @Schema(implementation = ExchangeHistorySimplifiedDto.class)
                )
        ),
        @ApiResponse(
                responseCode = "400",
                description = "잘못된 요청: 계좌를 찾을 수 없음",
                content = @Content(
                        mediaType = "application/json",
                        schema = @Schema(implementation = ErrorResponse.class),
                        examples = @ExampleObject(
                                name = "계좌를 찾을 수 없음",
                                summary = "사용자와 연관된 일반 계좌나 외화 계좌가 없는 경우",
                                value = "{\"status\": 400, \"httpError\": \"BAD_REQUEST\", \"errorCode\": \"ACCOUNT_NOT_FOUND\", \"message\": \"유저와 연관된 계좌를 찾을 수 없습니다.\"}"
                        )
                )
        ),
        @ApiResponse(
                responseCode = "401",
                description = "인증 실패 (JWT 또는 SID 누락/만료)",
                content = @Content(
                        mediaType = "application/json",
                        schema = @Schema(implementation = ErrorResponse.class),
                        examples = {
                                @ExampleObject(
                                        name = "유효하지 않은 액세스 토큰",
                                        value = "{\"status\":401, \"httpError\":\"UNAUTHORIZED\", \"errorCode\":\"INVALID_ACCESS_TOKEN\", \"message\":\"유효하지 않은 액세스 토큰입니다.\"}"
                                ),
                                @ExampleObject(
                                        name = "세션 ID 만료 또는 유효하지 않음",
                                        value = "{\"status\":401, \"httpError\":\"UNAUTHORIZED\", \"errorCode\":\"SID_INVALID_OR_EXPIRED\", \"message\":\"세션 ID가 유효하지 않거나 만료되었습니다.\"}"
                                )
                        }
                )
        ),
        @ApiResponse(
                responseCode = "500",
                description = "서버 내부 오류 또는 API 호출 실패",
                content = @Content(
                        mediaType = "application/json",
                        schema = @Schema(implementation = ErrorResponse.class),
                        examples = @ExampleObject(
                                name = "API 호출 실패",
                                value = "{\n  \"status\": 500,\n  \"httpError\": \"INTERNAL_SERVER_ERROR\",\n  \"errorCode\": \"API_CALL_FAILED\",\n  \"message\": \"신한 API 호출에 실패했습니다.\"\n}"
                        )
                )
        )
})
public @interface GetExchangeHistoryDocs {
}