package com.ssafy.ssashinsa.heyfy.inquire.docs;

import com.ssafy.ssashinsa.heyfy.account.dto.AccountNoDto;
import com.ssafy.ssashinsa.heyfy.common.exception.ErrorResponse;
import com.ssafy.ssashinsa.heyfy.inquire.dto.ForeignSingleDepositResponseDto; // DTO import
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.media.Content;
import io.swagger.v3.oas.annotations.media.ExampleObject;
import io.swagger.v3.oas.annotations.media.Schema;
import io.swagger.v3.oas.annotations.parameters.RequestBody;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import io.swagger.v3.oas.annotations.responses.ApiResponses;
import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

@Target(ElementType.METHOD)
@Retention(RetentionPolicy.RUNTIME)
@Operation(
        summary = "예금주 단일 외화 계좌 상세 조회",
        description = "단일 외화 계좌 정보를 조회합니다.",
        requestBody = @RequestBody(
                description = "조회할 외화 계좌번호",
                required = true,
                content = @Content(
                        mediaType = "application/json",
                        schema = @Schema(implementation = AccountNoDto.class),
                        examples = @ExampleObject(
                                name = "외화 계좌번호 요청 예시",
                                value = "{\n  \"accountNo\": \"0010475174188665\"\n}"
                        )
                )
        )
)
@ApiResponses({
        @ApiResponse(
                responseCode = "200",
                description = "성공적으로 외화 계좌 정보를 조회했습니다.",
                content = @Content(
                        mediaType = "application/json",
                        schema = @Schema(implementation = ForeignSingleDepositResponseDto.class),
                        examples = @ExampleObject(
                                name = "성공 응답 예시",
                                value = "{\n  \"bankName\": \"한국은행\",\n  \"userName\": \"test0820\",\n  \"accountNo\": \"0010475174188665\",\n  \"accountName\": \"한국은행 외화 수시입출금 상품\",\n  \"accountBalance\": \"0.00\",\n  \"currency\": \"USD\"\n}"
                        )
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
                                summary = "현재 유저의 DB에 요청된 계좌번호와 일치하는 계좌가 없는 경우",
                                value = "{\"status\": 400, \"httpError\": \"BAD_REQUEST\", \"errorCode\": \"ACCOUNT_NOT_FOUND\", \"message\": \"유저와 연관된 계좌를 찾을 수 없습니다.\"}"
                        )
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
public @interface InquireSingleForeignDepositDocs {
}