package com.ssafy.ssashinsa.heyfy.account.docs;

import com.ssafy.ssashinsa.heyfy.account.dto.AccountNoDto;
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
@Operation(
        summary = "계좌 인증 번호 확인",
        description = "사용자가 입력한 계좌와 인증 번호를 확인하고, 일치하면 계좌를 등록합니다."
)
@ApiResponses({
        @ApiResponse(
                responseCode = "200",
                description = "성공적으로 계좌 인증 및 등록 완료",
                content = @Content(
                        mediaType = "application/json",
                        schema = @Schema(implementation = AccountNoDto.class),
                        examples = @ExampleObject(
                                name = "성공 응답 예시",
                                value = "{\"accountNo\": \"0010756851096126\"}"
                        )
                )
        ),
        @ApiResponse(
                responseCode = "400",
                description = "잘못된 요청",
                content = @Content(
                        mediaType = "application/json",
                        schema = @Schema(implementation = ErrorResponse.class),
                        examples = {
                                @ExampleObject(
                                        name = "인증 번호 불일치",
                                        value = "{\"status\":400,\"httpError\":\"BAD_REQUEST\",\"errorCode\":\"FAIL_CHECK_AUTH\",\"message\":\"Authentication failed.\"}"
                                ),
                                @ExampleObject(
                                        name = "계좌 번호 유효성 실패",
                                        value = "{\"status\":400,\"httpError\":\"BAD_REQUEST\",\"errorCode\":\"A1003\",\"message\":\"Account number is not valid.\"}"
                                )
                        }
                )
        ),
        @ApiResponse(
                responseCode = "500",
                description = "서버 내부 오류",
                content = @Content(
                        mediaType = "application/json",
                        schema = @Schema(implementation = ErrorResponse.class),
                        examples = @ExampleObject(
                                name = "서버 내부 오류",
                                value = "{\"status\":500,\"httpError\":\"INTERNAL_SERVER_ERROR\",\"errorCode\":\"INTERNAL_SERVER_ERROR\",\"message\":\"An error has occurred.\"}"
                        )
                )
        )
})
public @interface AccountCheckDocs {
}