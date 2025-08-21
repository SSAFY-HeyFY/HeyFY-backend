package com.ssafy.ssashinsa.heyfy.register.docs;

import com.ssafy.ssashinsa.heyfy.common.exception.ErrorResponse;
import com.ssafy.ssashinsa.heyfy.register.dto.AccountCreationResponseDto; // 💡 이 DTO로 변경
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
@Operation(summary = "예금 계좌 등록", description = "신한은행 API를 이용하여 새로운 예금 계좌를 개설합니다.")
@ApiResponses({
        @ApiResponse(
                responseCode = "200",
                description = "성공적으로 계좌를 등록했습니다.",
                content = @Content(
                        mediaType = "application/json",
                        schema = @Schema(implementation = AccountCreationResponseDto.class),
                        examples = @ExampleObject(
                                name = "성공 응답 예시",
                                value = "{\"message\": \"정상처리 되었습니다.\", \"accountNo\": \"0016956302770649\"}"
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
                                        name = "유저키 누락",
                                        summary = "유저키가 존재하지 않는 경우",
                                        value = "{\"status\": 400, \"httpError\": \"BAD_REQUEST\", \"errorCode\": \"MISSING_USER_KEY\", \"message\": \"유저키가 누락되었습니다.\"}"
                                )
                        }
                )
        ),
        // 🔹 3. 실패 응답 (500 Internal Server Error)
        @ApiResponse(
                responseCode = "500",
                description = "서버 내부 오류",
                content = @Content(
                        mediaType = "application/json",
                        schema = @Schema(implementation = ErrorResponse.class),
                        examples = {
                                @ExampleObject(
                                        name = "유저를 찾을 수 없음",
                                        summary = "JWT 토큰의 사용자 정보를 찾을 수 없는 경우",
                                        value = "{\"status\": 500, \"httpError\": \"INTERNAL_SERVER_ERROR\", \"errorCode\": \"USER_NOT_FOUND\", \"message\": \"유저를 찾을 수 없습니다.\"}"
                                ),
                                @ExampleObject(
                                        name = "외부 API 호출 실패",
                                        summary = "신한은행 API 응답 오류 또는 서버 내부 오류",
                                        value = "{\"status\": 500, \"httpError\": \"INTERNAL_SERVER_ERROR\", \"errorCode\": \"API_CALL_FAILED\", \"message\": \"신한API 호출 에러\"}"
                                )
                        }
                )
        )
})
public @interface CreateDepositAccountDocs {
}