package com.ssafy.ssashinsa.heyfy.inquire.docs;

import com.ssafy.ssashinsa.heyfy.common.exception.ErrorResponse;
import com.ssafy.ssashinsa.heyfy.inquire.dto.ShinhanInquireDepositResponseDto;
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
@Tag(name = "Inquire", description = "계좌 조회 API")
@Operation(summary = "예금주 계좌 목록 조회", description = "사용자 키를 통해 연결된 계좌 목록을 조회합니다.")
@ApiResponses({
        @ApiResponse(
                responseCode = "200",
                description = "성공적으로 계좌 목록을 조회했습니다. (계좌가 없으면 REC 배열이 비어있음)",
                content = @Content(
                        mediaType = "application/json",
                        schema = @Schema(implementation = ShinhanInquireDepositResponseDto.class)
                )
        ),
        @ApiResponse(
                responseCode = "400",
                description = "잘못된 요청",
                content = @Content(
                        mediaType = "application/json",
                        schema = @Schema(implementation = ErrorResponse.class)
                )
        ),
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
public @interface InquireDepositListDocs {
}