package com.ssafy.ssashinsa.heyfy.account.docs;

import com.ssafy.ssashinsa.heyfy.account.dto.AccountAuthHttpResponseDto;
import com.ssafy.ssashinsa.heyfy.common.exception.ErrorResponse;
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
@Operation(summary = "1원 계좌 인증", description = "1원 이체로 계좌를 인증합니다.")
@ApiResponses({
        @ApiResponse(
                responseCode = "200",
                description = "1원 계좌 인증에 성공했습니다.",
                content = @Content(
                        mediaType = "application/json",
                        schema = @Schema(implementation = AccountAuthHttpResponseDto.class), // 💡 변경된 부분
                        examples = @ExampleObject(
                                name = "성공 응답 예시",
                                value = "{\"code\": \"1234\", \"accountNo\": \"110123456789\"}"
                        )
                )
        ),
        @ApiResponse(
                responseCode = "500",
                description = "서버 내부 오류 또는 인증 실패",
                content = @Content(
                        mediaType = "application/json",
                        schema = @Schema(implementation = ErrorResponse.class),
                        examples = {
                                @ExampleObject(
                                        name = "인증 실패",
                                        summary = "인증 번호 불일치 등 인증에 실패한 경우",
                                        value = "{\"status\": 500, \"httpError\": \"INTERNAL_SERVER_ERROR\", \"errorCode\": \"AUTH_FAILED\", \"message\": \"인증 실패\"}"
                                )
                        }
                )
        )
})
public @interface GetMyAccountAuthDocs {
}