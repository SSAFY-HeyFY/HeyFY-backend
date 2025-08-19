package com.ssafy.ssashinsa.heyfy.transfer.docs;

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
                        schema = @Schema(implementation = TransferHistoryResponse.class)
                )
        ),

        // 🔹 2. 실패 응답 - 잘못된 요청 (400 Bad Request)
        @ApiResponse(
                responseCode = "400",
                description = "잘못된 요청 (Bad Request)",
                content = @Content(
                        mediaType = "application/json",
                        schema = @Schema(implementation = ErrorResponse.class),
                        examples = {
                                @ExampleObject(
                                        name = "잘못된 계좌번호",
                                        summary = "A1003: 계좌번호 형식 오류",
                                        value = "{\"status\":400,\"httpError\":\"BAD_REQUEST\",\"errorCode\":\"INVALID_ACCOUNT_NO\",\"message\":\"입력하신 계좌 번호를 다시 확인해주세요.\"}"
                                ),
                                @ExampleObject(
                                        name = "잘못된 거래금액",
                                        summary = "A1011: 거래금액 형식 오류",
                                        value = "{\"status\":400,\"httpError\":\"BAD_REQUEST\",\"errorCode\":\"INVALID_TRANSACTION_AMOUNT\",\"message\":\"거래금액이 유효하지 않습니다.\"}"
                                ),
                                @ExampleObject(
                                        name = "계좌 잔액 부족",
                                        summary = "A1014: 잔액 부족",
                                        value = "{\"status\":422,\"httpError\":\"UNPROCESSABLE_ENTITY\",\"errorCode\":\"INSUFFICIENT_BALANCE\",\"message\":\"계좌잔액이 부족하여 거래가 실패했습니다.\"}"
                                ),
                                @ExampleObject(
                                        name = "1회 이체 한도 초과",
                                        summary = "A1016: 1회 이체 한도 초과",
                                        value = "{\"status\":422,\"httpError\":\"UNPROCESSABLE_ENTITY\",\"errorCode\":\"TRANSFER_LIMIT_EXCEEDED_ONCE\",\"message\":\"1회 이체가능한도를 초과했습니다.\"}"
                                )
                        }
                )
        )
})
public @interface TransferDocs {
}