package com.ssafy.ssashinsa.heyfy.register.docs;

import com.ssafy.ssashinsa.heyfy.account.dto.AccountNoDto;
import com.ssafy.ssashinsa.heyfy.common.exception.ErrorResponse;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.media.Content;
import io.swagger.v3.oas.annotations.media.ExampleObject;
import io.swagger.v3.oas.annotations.media.Schema;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import io.swagger.v3.oas.annotations.responses.ApiResponses;
import io.swagger.v3.oas.annotations.tags.Tag;
import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

@Target(ElementType.METHOD)
@Retention(RetentionPolicy.RUNTIME)
@Tag(name = "Register", description = "계좌 등록 API")
@Operation(summary = "일반 계좌 등록", description = "신한은행 API를 통해 기존의 일반 계좌를 확인하고 서비스에 등록합니다.")
@ApiResponses({
        @ApiResponse(
                responseCode = "200",
                description = "성공적으로 계좌를 등록했습니다.",
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
                description = "잘못된 요청 또는 계좌 유효성 오류",
                content = @Content(
                        mediaType = "application/json",
                        schema = @Schema(implementation = ErrorResponse.class),
                        examples = {
                                @ExampleObject(
                                        name = "계좌 유효성 실패",
                                        summary = "신한 API에서 계좌 번호가 유효하지 않다고 응답한 경우",
                                        value = "{\"status\":400,\"httpError\":\"BAD_REQUEST\",\"errorCode\":\"A1003\",\"message\":\"계좌번호가 유효하지 않습니다.\"}"
                                )
                        }
                )
        ),
        @ApiResponse(
                responseCode = "500",
                description = "서버 내부 오류 또는 신한 API 호출 오류",
                content = @Content(
                        mediaType = "application/json",
                        schema = @Schema(implementation = ErrorResponse.class)
                )
        )
})
public @interface RegisterAccountDocs {
}